#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:/Projects/W1z4rDV1510n/scripts/curate_noun_visual_audio_dictionary.py

License-Clean Multimodal Noun -> Canonical Image + Spoken Audio + Text Dictionary Curator
for W1z4rDV1510n.

Core Design Principles:
  1. Strict License Provenance:
     Fetches accurate encyclopedic & photographic noun images from Wikipedia / Wikimedia
     Commons (with Openverse fallback), accepting ONLY license-clean media (`CC0`,
     `Public Domain`, `CC-BY`, and `CC-BY-SA`, ranking CC0/PD/CC-BY highest and rejecting
     any NC, ND, Fair-Use, or unverified files). Full creator, source URL, and license
     URL provenance are recorded per noun.
  2. Spoken English Grounding Audio (C4rr13rX TTS Pipeline):
     Applies C4rr13rX's `ContextualLinguisticDisambiguator` heteronym pronunciation rules
     and synthesizes clean, un-embellished 16 kHz 16-bit mono PCM WAV spoken audio for
     both the isolated noun (`<slug>.wav`) and its concise canonical grounding sentence
     (`<slug>_sentence.wav`).
  3. Minimal-Neuron Canonical Decomposition (No Media Embellishment):
     Incoming images are downsampled (never upscaled or sharpened) to a compact 64x64
     canonical frame and decomposed into:
       - Exact `ImageBitsEncoder` tokens (`crates/core/src/streaming/image_bits.rs`)
       - Compact 8x8 quantized HSV+luminance cell grid (reconstructible with minimal neurons)
       - Strictly byte-disjoint shared visual primitives (`vcol:*`, `vshp:*`, `vedg:*`,
         `vlum:*`, `vsat:*`) so shared visual traits fire identical concept neurons across
         nouns for Hebbian cross-modal synthesis.
     Incoming 16 kHz mono WAV audio is silence-trimmed and decomposed into:
       - Exact `AudioBitsEncoder` tokens (`crates/core/src/streaming/audio_bits.rs`)
       - Compact 8x8 temporal-spectral nibble envelope
       - Strictly byte-disjoint shared acoustic/phonetic primitives (`adur:*`, `aspec:*`,
         `aons:*`, `avow:*`, `acod:*`).
  4. Cloud Upload -> Remote Hash/Size Verify -> Immediate Local `unlink()`:
     Uses `MultimodalCloudSync` (`scripts/multimodal_cloud_sync.py`) to upload each
     canonical PNG and WAV to cloud storage, verify remote existence + byte size + MD5/SHA256,
     and immediately `unlink()` local temporary files so zero media accumulates on disk.
  5. Polite Elastic PC-Allowance ("Nice" Resource Scaling):
     Runs at `psutil.BELOW_NORMAL_PRIORITY_CLASS`, keeps >= 8.0 GB of system RAM untouched
     (+ 12% under-cap cushion so concurrent crypto trading strategies never feel squeezed),
     scales batch sizes UP when available RAM >= 14.5 GB, and shrinks batch sizes to 1 +
     `gc.collect()` when RAM tightens without halting curation.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import html
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Allow seamless imports from CoolCryptoUtilities venv & W1z4rDV1510n scripts
_CCU_ROOT = Path("D:/Projects/CoolCryptoUtilities")
_CCU_VENV_SITE = _CCU_ROOT / ".venv" / "Lib" / "site-packages"
_W1Z4RD_ROOT = Path("D:/Projects/W1z4rDV1510n")
_SCRIPTS_DIR = _W1Z4RD_ROOT / "scripts"
_C4RR13RX_SCRIPTS = Path("D:/Projects/C4rr13rX/scripts")

for _p in (_SCRIPTS_DIR, _CCU_VENV_SITE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import psutil
import requests
from PIL import Image

from multimodal_cloud_sync import MultimodalCloudSync, load_env_file

# Hydrate non-secret env defaults if present (read-only)
if (_CCU_ROOT / ".env").exists():
    load_env_file(_CCU_ROOT / ".env", override=False)

# Import C4rr13rX ContextualLinguisticDisambiguator when available
try:
    if _C4RR13RX_SCRIPTS.exists() and str(_C4RR13RX_SCRIPTS) not in sys.path:
        sys.path.append(str(_C4RR13RX_SCRIPTS))
    from natural_tts_pipeline import ContextualLinguisticDisambiguator  # type: ignore
except Exception:  # noqa: BLE001
    class ContextualLinguisticDisambiguator:  # type: ignore[no-redef]
        @classmethod
        def disambiguate(cls, text: str, lang: str = "en") -> str:
            working = text
            if re.search(r"\blead\s+(pipe|weight|metal|block)\b", working, re.I):
                working = re.sub(r"\blead\b", "led", working, flags=re.I)
            if re.search(r"\b(the|strong|cold|moving)\s+wind\b", working, re.I):
                working = re.sub(r"\bwind\b", "wihnd", working, flags=re.I)
            return working


USER_AGENT = (
    "W1z4rDV1510n-NounMultimodalCurator/1.0 "
    "(https://github.com/C4rr13rX/W1z4rDV1510n; educational multimodal Hebbian dictionary)"
)
WIKI_API = "https://en.wikipedia.org/w/api.php"
SIMPLE_WIKI_API = "https://simple.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"

DEFAULT_OUT_DIR = _W1Z4RD_ROOT / "data" / "multimodal_noun_dictionary"
DEFAULT_MANIFEST = DEFAULT_OUT_DIR / "noun_multimodal_dictionary.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUT_DIR / "noun_multimodal_summary.json"


# ============================================================================
# 1. Polite Elastic PC-Allowance Governor ("Nice" on RAM & CPU)
# ============================================================================

def _trim_current_process_working_set() -> bool:
    if os.name != "nt":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_bool
        return bool(psapi.EmptyWorkingSet(handle))
    except Exception:  # noqa: BLE001
        return False


class PoliteResourceGovernor:
    """Elastic RAM & CPU governor keeping >= 8.0 GB + 12% cushion untouched."""

    RESERVE_RAM_GB: float = 8.0
    UNDER_CAP_CUSHION: float = 0.12
    # Leaving >= 8.0 GB untouched AFTER applying a 12% under-cap safety cushion:
    # 8.0 / (1.0 - 0.12) = 9.0909 GB effective soft floor.
    EFFECTIVE_FLOOR_GB: float = round(8.0 / (1.0 - 0.12), 3)
    SCALE_UP_RAM_GB: float = 14.5

    def __init__(self, max_batch_size: int = 16) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self.proc = psutil.Process(os.getpid())
        self.priority_applied = self._set_below_normal_priority()
        self.min_avail_gb: float = 999.0
        self.max_avail_gb: float = 0.0
        self.max_rss_mb: float = 0.0
        self.scale_up_steps: int = 0
        self.tight_shrink_steps: int = 0
        self.gc_sweeps: int = 0

    def _set_below_normal_priority(self) -> str:
        try:
            if os.name == "nt" and hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                self.proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                return "BELOW_NORMAL_PRIORITY_CLASS"
            self.proc.nice(10)
            return "nice_10"
        except Exception as exc:  # noqa: BLE001
            return f"default ({exc})"

    def sample(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024.0 ** 3)
        total_gb = vm.total / (1024.0 ** 3)
        rss_mb = self.proc.memory_info().rss / (1024.0 ** 2)

        self.min_avail_gb = min(self.min_avail_gb, avail_gb)
        self.max_avail_gb = max(self.max_avail_gb, avail_gb)
        self.max_rss_mb = max(self.max_rss_mb, rss_mb)

        # Usable allowance above 8.0 GB discounted by 12% under-cap cushion
        usable_allowance_gb = max(0.0, (avail_gb - self.RESERVE_RAM_GB) * (1.0 - self.UNDER_CAP_CUSHION))

        if avail_gb >= self.SCALE_UP_RAM_GB:
            batch_size = self.max_batch_size
            mode = "scale_up_abundant"
            pace_sleep = 0.01
            self.scale_up_steps += 1
        elif avail_gb >= 11.5:
            batch_size = min(self.max_batch_size, 6)
            mode = "balanced_mid"
            pace_sleep = 0.03
        elif avail_gb >= self.EFFECTIVE_FLOOR_GB:
            batch_size = min(self.max_batch_size, 2)
            mode = "polite_guarded"
            pace_sleep = 0.05
        else:
            # RAM tightened inside the 8.0 GB + 12% cushion band:
            # Shrink batch size to 1, sweep GC + trim working set, continue smoothly!
            batch_size = 1
            mode = "tight_cushioned_batch1"
            pace_sleep = 0.12
            self.tight_shrink_steps += 1
            self.sweep_memory()

        return {
            "total_gb": round(total_gb, 2),
            "avail_gb": round(avail_gb, 2),
            "usable_allowance_gb": round(usable_allowance_gb, 2),
            "rss_mb": round(rss_mb, 1),
            "batch_size": batch_size,
            "mode": mode,
            "pace_sleep": pace_sleep,
        }

    def sweep_memory(self) -> None:
        gc.collect()
        _trim_current_process_working_set()
        self.gc_sweeps += 1

    def summary(self) -> dict[str, Any]:
        return {
            "priority_class": self.priority_applied,
            "reserve_ram_gb": self.RESERVE_RAM_GB,
            "under_cap_cushion_pct": int(self.UNDER_CAP_CUSHION * 100),
            "effective_floor_gb": self.EFFECTIVE_FLOOR_GB,
            "scale_up_threshold_gb": self.SCALE_UP_RAM_GB,
            "min_available_ram_gb": round(self.min_avail_gb if self.min_avail_gb < 900 else 0.0, 2),
            "max_available_ram_gb": round(self.max_avail_gb, 2),
            "peak_process_rss_mb": round(self.max_rss_mb, 1),
            "scale_up_batches": self.scale_up_steps,
            "tight_batch1_adjustments": self.tight_shrink_steps,
            "gc_working_set_sweeps": self.gc_sweeps,
        }


# ============================================================================
# 2. Concrete Noun Catalog + W1z4rD Corpus Extractor
# ============================================================================

@dataclass(frozen=True)
class NounSpec:
    noun: str
    domain: str
    category: str
    affordance: str
    wiki_title: str
    grounding_sentence: str
    corpus_source: str = "curated_core"


# High-precision core concrete nouns spanning robotics, materials, nature,
# biodiversity, anatomy/medical, STEM instruments, food, and everyday objects.
CORE_CONCRETE_NOUNS: list[NounSpec] = [
    # Robotics, Mechanics & Engineering
    NounSpec("gear", "robotics_mechanics", "machine_element", "transmit_torque", "Gear", "A gear is a rotating toothed wheel that transmits mechanical motion."),
    NounSpec("motor", "robotics_mechanics", "actuator_device", "convert_energy", "Electric motor", "A motor converts electrical energy into mechanical rotation."),
    NounSpec("wheel", "robotics_mechanics", "machine_element", "roll_transport", "Wheel", "A wheel is a circular block that rotates on an axle for smooth motion."),
    NounSpec("battery", "robotics_mechanics", "power_source", "store_charge", "Electric battery", "A battery stores chemical energy and supplies electrical power."),
    NounSpec("chassis", "robotics_mechanics", "structural_frame", "support_body", "Chassis", "A chassis is the load-bearing frame that supports a vehicle or robot."),
    NounSpec("lever", "robotics_mechanics", "machine_element", "amplify_force", "Lever", "A lever is a rigid bar resting on a pivot used to multiply force."),
    NounSpec("pulley", "robotics_mechanics", "machine_element", "lift_load", "Pulley", "A pulley is a grooved wheel on an axle that guides a rope or belt."),
    NounSpec("axle", "robotics_mechanics", "machine_element", "support_rotation", "Axle", "An axle is a central shaft for a rotating wheel or gear."),
    NounSpec("spring", "robotics_mechanics", "elastic_element", "store_elasticity", "Spring (device)", "A mechanical spring is an elastic coil that stores mechanical energy."),
    NounSpec("bearing", "robotics_mechanics", "machine_element", "reduce_friction", "Bearing (mechanical)", "A bearing constrains motion and reduces friction between moving parts."),
    NounSpec("piston", "robotics_mechanics", "machine_element", "compress_fluid", "Piston", "A piston is a moving disk inside a cylinder that transfers force."),
    NounSpec("valve", "robotics_mechanics", "flow_device", "regulate_flow", "Valve", "A valve regulates or directs the flow of a fluid through a passage."),
    NounSpec("circuit", "robotics_mechanics", "electronic_system", "route_current", "Electronic circuit", "An electronic circuit connects components so electric current can perform work."),
    NounSpec("sensor", "robotics_mechanics", "measurement_device", "detect_signal", "Sensor", "A sensor detects physical changes and converts them into measurable signals."),
    NounSpec("magnet", "robotics_mechanics", "field_object", "attract_iron", "Magnet", "A magnet produces a magnetic field that attracts ferromagnetic metals."),
    NounSpec("bolt", "robotics_mechanics", "fastener_part", "fasten_joint", "Bolt (fastener)", "A bolt is a threaded metal fastener paired with a nut to join parts."),
    NounSpec("screw", "robotics_mechanics", "fastener_part", "fasten_joint", "Screw", "A screw converts rotational torque into linear fastening force."),
    NounSpec("wrench", "robotics_mechanics", "hand_tool", "apply_torque", "Wrench", "A wrench grips and turns nuts and bolts with mechanical leverage."),
    NounSpec("hammer", "robotics_mechanics", "hand_tool", "strike_impact", "Hammer", "A hammer delivers a concentrated impact force to drive fasteners or shape metal."),
    NounSpec("anvil", "robotics_mechanics", "metalwork_tool", "support_forging", "Anvil", "An anvil is a heavy steel block on which heated metal is forged."),

    # Materials & Matter
    NounSpec("water", "materials_matter", "liquid_substance", "hydrate_flow", "Water", "Water is a clear liquid compound of hydrogen and oxygen essential for life."),
    NounSpec("ice", "materials_matter", "solid_crystal", "cool_freeze", "Ice", "Ice is water frozen into a rigid crystalline solid state."),
    NounSpec("fire", "materials_matter", "combustion_plasma", "radiate_heat", "Fire", "Fire is rapid oxidation releasing heat, flame, and light."),
    NounSpec("steel", "materials_matter", "metal_alloy", "build_structure", "Steel", "Steel is a strong alloy of iron and carbon used in tools and structures."),
    NounSpec("copper", "materials_matter", "conductive_metal", "conduct_current", "Copper", "Copper is a ductile reddish metal with high electrical and thermal conductivity."),
    NounSpec("gold", "materials_matter", "precious_metal", "conduct_preserve", "Gold", "Gold is a dense yellow metal resistant to corrosion and oxidation."),
    NounSpec("glass", "materials_matter", "amorphous_solid", "transmit_light", "Glass", "Glass is a hard transparent solid used for windows, lenses, and vessels."),
    NounSpec("wood", "materials_matter", "organic_material", "build_structure", "Wood", "Wood is the fibrous structural tissue found in the stems and roots of trees."),
    NounSpec("rubber", "materials_matter", "elastic_polymer", "absorb_shock", "Natural rubber", "Rubber is a flexible elastic polymer capable of stretching and rebounding."),
    NounSpec("silicon", "materials_matter", "semiconductor_solid", "switch_electrons", "Silicon", "Silicon is a crystalline metalloid semiconductor used in integrated circuits."),
    NounSpec("rock", "materials_matter", "mineral_aggregate", "build_foundation", "Rock (geology)", "A rock is a naturally occurring solid aggregate of one or more minerals."),
    NounSpec("sand", "materials_matter", "granular_matter", "filter_pack", "Sand", "Sand is a granular material composed of finely divided mineral particles."),
    NounSpec("crystal", "materials_matter", "ordered_lattice", "diffract_light", "Crystal", "A crystal is a solid whose atoms are arranged in a repeating geometric lattice."),
    NounSpec("diamond", "materials_matter", "carbon_crystal", "cut_abrade", "Diamond", "Diamond is an ultra-hard transparent crystal of pure carbon."),

    # Nature, Earth & Astronomy
    NounSpec("tree", "nature_earth", "woody_plant", "photosynthesize", "Tree", "A tree is a tall perennial plant with an elongated woody trunk and branches."),
    NounSpec("leaf", "nature_earth", "plant_organ", "photosynthesize", "Leaf", "A leaf is a flat green plant organ specialized for photosynthesis."),
    NounSpec("flower", "nature_earth", "reproductive_bloom", "pollinate_seed", "Flower", "A flower is the colorful reproductive structure of flowering plants."),
    NounSpec("seed", "nature_earth", "plant_embryo", "germinate_grow", "Seed", "A seed is an embryonic plant enclosed in a protective outer covering."),
    NounSpec("sun", "nature_earth", "stellar_body", "radiate_light", "Sun", "The Sun is the star at the center of the Solar System that warms Earth."),
    NounSpec("moon", "nature_earth", "natural_satellite", "orbit_planet", "Moon", "The Moon is Earth's natural rocky satellite visible by reflected sunlight."),
    NounSpec("cloud", "nature_earth", "atmospheric_vapor", "precipitate_rain", "Cloud", "A cloud is a visible aerosol mass of tiny liquid droplets in the sky."),
    NounSpec("lightning", "nature_earth", "electric_discharge", "discharge_charge", "Lightning", "Lightning is a giant electrostatic discharge during a thunderstorm."),
    NounSpec("rainbow", "nature_earth", "optical_spectrum", "disperse_light", "Rainbow", "A rainbow is a multicolored arc formed by refraction of light in water droplets."),
    NounSpec("mountain", "nature_earth", "elevated_landform", "elevate_terrain", "Mountain", "A mountain is a large landform rising prominently above surrounding terrain."),
    NounSpec("volcano", "nature_earth", "geologic_vent", "erupt_magma", "Volcano", "A volcano is a rupture in a planet's crust that allows hot lava and ash to escape."),
    NounSpec("river", "nature_earth", "flowing_waterway", "channel_water", "River", "A river is a natural flowing watercourse moving toward an ocean or lake."),
    NounSpec("ocean", "nature_earth", "marine_basin", "sustain_marine", "Ocean", "An ocean is a vast continuous body of salt water covering most of Earth."),
    NounSpec("glacier", "nature_earth", "ice_mass", "carve_valley", "Glacier", "A glacier is a persistent body of dense ice slowly moving under its own weight."),

    # Biodiversity & Animals
    NounSpec("dog", "biodiversity_fauna", "domestic_mammal", "companion_guard", "Dog", "A dog is a domesticated carnivorous mammal known for loyalty and keen senses."),
    NounSpec("cat", "biodiversity_fauna", "domestic_mammal", "stalk_climb", "Cat", "A cat is a small agile domesticated feline mammal with retractable claws."),
    NounSpec("horse", "biodiversity_fauna", "hoofed_mammal", "gallop_carry", "Horse", "A horse is a large hoofed herbivorous mammal domesticated for riding and draft."),
    NounSpec("cow", "biodiversity_fauna", "hoofed_mammal", "graze_lactate", "Cattle", "A cow is a large domesticated ungulate raised for milk and grazing."),
    NounSpec("elephant", "biodiversity_fauna", "trunked_mammal", "grasp_forage", "Elephant", "An elephant is the largest living land mammal, having a muscular trunk and tusks."),
    NounSpec("bear", "biodiversity_fauna", "ursid_mammal", "forage_climb", "Bear", "A bear is a large stocky carnivoran mammal with thick fur and strong limbs."),
    NounSpec("lion", "biodiversity_fauna", "felid_mammal", "hunt_pack", "Lion", "A lion is a large social wild cat native to grasslands and savannas."),
    NounSpec("wolf", "biodiversity_fauna", "canid_mammal", "hunt_pack", "Wolf", "A wolf is a social wild canine native to wilderness and tundra habitats."),
    NounSpec("rabbit", "biodiversity_fauna", "lagomorph_mammal", "hop_burrow", "Rabbit", "A rabbit is a small herbivorous mammal with long ears and powerful hind legs."),
    NounSpec("bird", "biodiversity_fauna", "feathered_vertebrate", "fly_perch", "Bird", "A bird is a warm-blooded feathered vertebrate with wings and a beak."),
    NounSpec("eagle", "biodiversity_fauna", "raptor_bird", "soar_hunt", "Eagle", "An eagle is a large bird of prey with keen vision and curved talons."),
    NounSpec("owl", "biodiversity_fauna", "nocturnal_bird", "hunt_silent", "Owl", "An owl is a nocturnal bird of prey with forward-facing eyes and silent flight."),
    NounSpec("penguin", "biodiversity_fauna", "aquatic_bird", "swim_dive", "Penguin", "A penguin is a flightless aquatic bird adapted for swimming in cold seas."),
    NounSpec("fish", "biodiversity_fauna", "gilled_vertebrate", "swim_gill", "Fish", "A fish is an aquatic gill-bearing vertebrate with fins for swimming."),
    NounSpec("whale", "biodiversity_fauna", "marine_mammal", "swim_breach", "Whale", "A whale is a large fully aquatic marine mammal that breathes air through a blowhole."),
    NounSpec("octopus", "biodiversity_fauna", "cephalopod_mollusc", "grasp_camouflage", "Octopus", "An octopus is a soft-bodied eight-armed marine mollusc with high intelligence."),
    NounSpec("turtle", "biodiversity_fauna", "shelled_reptile", "shield_swim", "Turtle", "A turtle is a reptile protected by a bony shell developed from its ribs."),
    NounSpec("frog", "biodiversity_fauna", "tailless_amphibian", "leap_swim", "Frog", "A frog is a short-bodied amphibian with smooth moist skin and jumping legs."),
    NounSpec("bee", "biodiversity_fauna", "winged_insect", "pollinate_hive", "Bee", "A bee is a winged pollinating insect known for producing wax and honey."),
    NounSpec("butterfly", "biodiversity_fauna", "winged_insect", "pollinate_flutter", "Butterfly", "A butterfly is a diurnal insect with broad brightly patterned wings."),
    NounSpec("ant", "biodiversity_fauna", "eusocial_insect", "forage_colony", "Ant", "An ant is a social insect that lives in organized underground or mound colonies."),
    NounSpec("spider", "biodiversity_fauna", "eight_legged_arachnid", "spin_web", "Spider", "A spider is an air-breathing eight-legged arachnid that spins silk."),
    NounSpec("snail", "biodiversity_fauna", "shelled_gastropod", "glide_shell", "Snail", "A snail is a shelled gastropod mollusc that moves on a muscular foot."),
    NounSpec("mushroom", "biodiversity_fauna", "fungal_fruiting_body", "disperse_spores", "Mushroom", "A mushroom is the fleshy spore-bearing fruiting body of a fungus."),
    NounSpec("coral", "biodiversity_fauna", "colonial_cnidarian", "build_reef", "Coral", "Coral consists of colonial marine invertebrates that build calcium carbonate reefs."),

    # Food & Edible Produce
    NounSpec("apple", "food_produce", "pome_fruit", "eat_nourish", "Apple", "An apple is a crisp round edible fruit produced by an apple tree."),
    NounSpec("banana", "food_produce", "elongated_fruit", "peel_eat", "Banana", "A banana is an elongated curved edible fruit with a soft starchy flesh."),
    NounSpec("orange", "food_produce", "citrus_fruit", "juice_eat", "Orange (fruit)", "An orange is a round juicy citrus fruit rich in vitamin C."),
    NounSpec("strawberry", "food_produce", "berry_fruit", "eat_nourish", "Strawberry", "A strawberry is a bright red sweet fruit studded with tiny surface seeds."),
    NounSpec("lemon", "food_produce", "citrus_fruit", "flavor_acid", "Lemon", "A lemon is a yellow ellipsoidal citrus fruit with sour acidic juice."),
    NounSpec("carrot", "food_produce", "root_vegetable", "eat_nourish", "Carrot", "A carrot is a crisp orange root vegetable grown in soil."),
    NounSpec("tomato", "food_produce", "edible_berry", "cook_eat", "Tomato", "A tomato is a glossy red edible fruit widely used in cooking."),
    NounSpec("potato", "food_produce", "starchy_tuber", "cook_eat", "Potato", "A potato is a starchy underground stem tuber grown as a staple food."),
    NounSpec("bread", "food_produce", "baked_grain", "slice_eat", "Bread", "Bread is a staple food prepared by baking a dough of flour and water."),
    NounSpec("egg", "food_produce", "shelled_ovum", "cook_nourish", "Egg as food", "An egg is an oval shelled food rich in protein laid by birds."),
    NounSpec("cheese", "food_produce", "dairy_food", "slice_eat", "Cheese", "Cheese is a solid dairy food made from pressed curds of milk protein."),

    # Anatomy & Medical
    NounSpec("hand", "anatomy_medical", "body_extremity", "grasp_manipulate", "Hand", "A hand is a prehensile multi-fingered organ at the end of the forearm."),
    NounSpec("eye", "anatomy_medical", "sensory_organ", "focus_vision", "Human eye", "An eye is an optical sensory organ that detects light and forms visual images."),
    NounSpec("ear", "anatomy_medical", "sensory_organ", "detect_sound", "Ear", "An ear is the organ that enables hearing and body balance."),
    NounSpec("heart", "anatomy_medical", "muscular_organ", "pump_blood", "Heart", "The heart is a muscular organ that pumps blood through the circulatory vessels."),
    NounSpec("brain", "anatomy_medical", "neural_organ", "process_cognition", "Human brain", "The brain is the central organ of the nervous system controlling thought and action."),
    NounSpec("lung", "anatomy_medical", "respiratory_organ", "exchange_gas", "Lung", "The lungs are spongy respiratory organs that exchange oxygen and carbon dioxide."),
    NounSpec("skull", "anatomy_medical", "skeletal_structure", "protect_brain", "Skull", "The skull is a bony structure that forms the head and protects the brain."),
    NounSpec("bone", "anatomy_medical", "rigid_tissue", "support_skeleton", "Bone", "A bone is a rigid mineralized organ that forms part of the skeleton."),
    NounSpec("tooth", "anatomy_medical", "calcified_organ", "chew_bite", "Human tooth", "A tooth is a hard calcified structure in the jaw used to break down food."),
    NounSpec("cell", "anatomy_medical", "biological_unit", "metabolize_divide", "Cell (biology)", "A cell is the basic membrane-bound structural and functional unit of life."),
    NounSpec("neuron", "anatomy_medical", "excitable_cell", "transmit_spike", "Neuron", "A neuron is an electrically excitable nerve cell that communicates via synapses."),
    NounSpec("stethoscope", "anatomy_medical", "clinical_instrument", "auscultate_sound", "Stethoscope", "A stethoscope is an acoustic medical device for listening to internal body sounds."),

    # STEM Instruments & Optics
    NounSpec("lens", "stem_instruments", "optical_element", "refract_focus", "Lens", "A lens is a transmissive optical device that focuses or disperses light beams."),
    NounSpec("prism", "stem_instruments", "optical_element", "disperse_spectrum", "Prism (optics)", "An optical prism is a transparent flat-polished element that refracts light."),
    NounSpec("microscope", "stem_instruments", "scientific_instrument", "magnify_tiny", "Microscope", "A microscope is a laboratory instrument used to examine objects too small for the naked eye."),
    NounSpec("telescope", "stem_instruments", "scientific_instrument", "observe_distant", "Telescope", "A telescope collects electromagnetic radiation to observe distant celestial objects."),
    NounSpec("compass", "stem_instruments", "navigation_instrument", "indicate_north", "Compass", "A compass is a navigational instrument that shows direction relative to magnetic north."),
    NounSpec("thermometer", "stem_instruments", "measurement_device", "measure_temp", "Thermometer", "A thermometer is a device that measures temperature or a temperature gradient."),
    NounSpec("clock", "stem_instruments", "timekeeping_device", "measure_time", "Clock", "A clock is a mechanical or electronic instrument that measures and displays time."),
    NounSpec("abacus", "stem_instruments", "counting_frame", "calculate_beads", "Abacus", "An abacus is a manual calculating frame with sliding beads on rods."),

    # Everyday Artifacts, Architecture & Vehicles
    NounSpec("book", "artifacts_vehicles", "printed_volume", "record_knowledge", "Book", "A book is a bound medium of pages used to record information and stories."),
    NounSpec("key", "artifacts_vehicles", "security_tool", "actuate_lock", "Key (lock)", "A key is a shaped metal instrument used to operate a mechanical lock."),
    NounSpec("lock", "artifacts_vehicles", "security_device", "secure_closure", "Lock (security device)", "A lock is a mechanical fastening device released by a key or combination."),
    NounSpec("bell", "artifacts_vehicles", "percussion_resonator", "ring_chime", "Bell", "A bell is a hollow flared resonator that emits a clear musical tone when struck."),
    NounSpec("drum", "artifacts_vehicles", "percussion_instrument", "beat_rhythm", "Drum", "A drum is a cylindrical percussion instrument sounded by striking a stretched membrane."),
    NounSpec("violin", "artifacts_vehicles", "string_instrument", "bow_melody", "Violin", "A violin is a wooden four-stringed musical instrument played with a horsehair bow."),
    NounSpec("chair", "artifacts_vehicles", "seating_furniture", "support_sitting", "Chair", "A chair is a raised piece of furniture with a backrest designed for one person to sit."),
    NounSpec("table", "artifacts_vehicles", "flat_furniture", "support_objects", "Table (furniture)", "A table is an item of furniture with a flat horizontal top supported by legs."),
    NounSpec("lamp", "artifacts_vehicles", "lighting_fixture", "illuminate_space", "Lamp", "A lamp is a fixture that produces artificial light from an electric bulb or flame."),
    NounSpec("cup", "artifacts_vehicles", "drinking_vessel", "hold_liquid", "Cup", "A cup is a small open container used for holding and drinking liquids."),
    NounSpec("bottle", "artifacts_vehicles", "narrow_neck_vessel", "contain_liquid", "Bottle", "A bottle is a narrow-necked rigid or semi-rigid container that stores liquids."),
    NounSpec("umbrella", "artifacts_vehicles", "folding_canopy", "shield_rain", "Umbrella", "An umbrella is a folding canopy supported by ribs that shields against rain or sun."),
    NounSpec("scissors", "artifacts_vehicles", "pivoted_blades", "shear_material", "Scissors", "Scissors are hand-operated shearing blades pivoted together to cut thin materials."),
    NounSpec("bridge", "artifacts_vehicles", "spanning_structure", "span_obstacle", "Bridge", "A bridge is a built structure spanning a river, canyon, or road for passage."),
    NounSpec("lighthouse", "artifacts_vehicles", "coastal_tower", "guide_ships", "Lighthouse", "A lighthouse is a coastal tower emitting light from lenses to guide maritime ships."),
    NounSpec("bicycle", "artifacts_vehicles", "human_powered_vehicle", "pedal_ride", "Bicycle", "A bicycle is a two-wheeled steerable vehicle propelled by human pedals."),
    NounSpec("car", "artifacts_vehicles", "motor_vehicle", "drive_road", "Car", "A car is a wheeled motor vehicle used for transporting passengers on roads."),
    NounSpec("train", "artifacts_vehicles", "rail_vehicle", "haul_rail", "Train", "A train is a connected series of rail vehicles running along a railway track."),
    NounSpec("airplane", "artifacts_vehicles", "fixed_wing_aircraft", "fly_transport", "Airplane", "An airplane is a fixed-wing aircraft propelled forward by thrust from engines."),
    NounSpec("boat", "artifacts_vehicles", "watercraft_vessel", "float_navigate", "Boat", "A boat is a buoyant watercraft designed to float and travel across water."),
    NounSpec("submarine", "artifacts_vehicles", "submersible_vessel", "dive_underwater", "Submarine", "A submarine is a watertight watercraft capable of independent operation underwater."),
    NounSpec("rocket", "artifacts_vehicles", "reaction_vehicle", "launch_orbit", "Rocket", "A rocket is a vehicle propelled by high-velocity exhaust from a rocket engine."),
]

# Exclusion list when harvesting additional nouns from developmental / K-12 corpuses
_NON_CONCRETE_STOPWORDS = {
    "no", "yes", "hi", "bye", "please", "sorry", "okay", "hello", "ow", "yay", "wow", "shh",
    "more", "up", "down", "go", "stop", "come", "look", "see", "get", "give", "want", "open",
    "eat", "drink", "hot", "cold", "big", "little", "mine", "wet", "dirty", "small", "tall",
    "short", "long", "wide", "thin", "thick", "heavy", "warm", "cool", "fast", "slow", "loud",
    "quiet", "hard", "soft", "rough", "smooth", "sharp", "flat", "round", "straight", "curved",
    "clean", "dry", "new", "old", "broken", "closed", "happy", "sad", "angry", "scared",
    "surprised", "excited", "tired", "hungry", "thirsty", "sick", "hurt", "bored", "love",
    "hate", "afraid", "shy", "proud", "confused", "red", "blue", "green", "yellow", "purple",
    "pink", "brown", "black", "white", "gray", "silver", "turquoise", "beige", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero", "hundred",
    "thousand", "half", "many", "few", "empty", "full", "addition", "subtraction",
    "multiplication", "division", "africa", "antarctica", "asia", "australia", "europe",
    "vote", "word", "alphabet", "congress", "constitution", "training_log",
}

_CONCRETE_CATEGORIES = {
    "animal", "food", "vehicle", "clothing", "home", "nature", "body", "toy", "shape", "place",
}


def build_noun_catalog() -> list[NounSpec]:
    """Combine curated multi-domain concrete nouns with W1z4rD corpus nouns."""
    seen: set[str] = set()
    catalog: list[NounSpec] = []

    for spec in CORE_CONCRETE_NOUNS:
        key = spec.noun.strip().lower()
        if key not in seen:
            seen.add(key)
            catalog.append(spec)

    # Harvest additional concrete nouns from toddler_v2, k12, and concept_dataset.jsonl
    toddler_dir = _W1Z4RD_ROOT / "data" / "multimodal_toddler_v2"
    if toddler_dir.exists():
        for jpg in sorted(toddler_dir.glob("*.jpg")):
            word = jpg.stem.strip().lower().replace("_", " ")
            if word in seen or word in _NON_CONCRETE_STOPWORDS or not word.isalpha():
                continue
            seen.add(word)
            catalog.append(NounSpec(
                noun=word,
                domain="toddler_corpus",
                category="concrete_object",
                affordance="observe_interact",
                wiki_title=word.capitalize(),
                grounding_sentence=f"A {word} is a concrete everyday object or organism.",
                corpus_source="multimodal_toddler_v2",
            ))

    k12_dir = _W1Z4RD_ROOT / "data" / "multimodal_k12"
    if k12_dir.exists():
        for jpg in sorted(k12_dir.glob("*.jpg")):
            word = jpg.stem.strip().lower()
            if "_" in word or word in seen or word in _NON_CONCRETE_STOPWORDS or not word.isalpha():
                continue
            seen.add(word)
            catalog.append(NounSpec(
                noun=word,
                domain="k12_corpus",
                category="k12_noun",
                affordance="study_observe",
                wiki_title=word.capitalize(),
                grounding_sentence=f"A {word} is a physical entity studied in science and everyday life.",
                corpus_source="multimodal_k12",
            ))

    concept_file = _W1Z4RD_ROOT / "data" / "foundation" / "concept_dataset.jsonl"
    if concept_file.exists():
        with concept_file.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                cat = str(rec.get("category") or "").strip().lower()
                clean = str(rec.get("clean") or rec.get("concept") or "").strip().lower()
                clean = re.sub(r"\s*\([^)]*\)\s*", "", clean).strip()
                if (
                    not clean
                    or clean in seen
                    or clean in _NON_CONCRETE_STOPWORDS
                    or cat not in _CONCRETE_CATEGORIES
                    or not re.fullmatch(r"[a-z]+(?: [a-z]+)?", clean)
                ):
                    continue
                defn = str(rec.get("definition") or "").strip()
                first_sent = re.split(r"(?<=[.!?])\s+", defn)[0].strip() if defn else ""
                if not first_sent or len(first_sent) > 140:
                    first_sent = f"A {clean} is a concrete {cat} in the physical world."
                seen.add(clean)
                catalog.append(NounSpec(
                    noun=clean,
                    domain=f"foundation_{cat}",
                    category=cat,
                    affordance=f"ground_{cat}",
                    wiki_title=str(rec.get("wiki_title") or clean.capitalize()),
                    grounding_sentence=first_sent,
                    corpus_source="foundation/concept_dataset.jsonl",
                ))

    return catalog


# ============================================================================
# 3. Strict License Gate & Encyclopedic Image Fetcher
# ============================================================================

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_html_meta(val: Any) -> str:
    if isinstance(val, dict):
        val = val.get("value", "")
    text = html.unescape(_HTML_TAG_RE.sub(" ", str(val or "")))
    return re.sub(r"\s+", " ", text).strip()


def classify_clean_license(
    short_name: str,
    usage_terms: str,
    license_url: str,
    nonfree: str = "",
    *,
    allow_sa: bool = True,
) -> Optional[tuple[str, str, int]]:
    """Return `(normalized_license, license_family, priority_rank)` if strictly allowed, else None.

    Allowed families ONLY:
      rank 4: CC0
      rank 3: Public Domain
      rank 2: CC-BY
      rank 1: CC-BY-SA (when allow_sa=True)
    Explicitly rejects any NC, ND, Fair Use, Non-free, GFDL-only, or unknown license.
    """
    if str(nonfree).strip().lower() in {"1", "true", "yes"}:
        return None

    combined = f"{short_name} | {usage_terms} | {license_url}".lower()
    if any(bad in combined for bad in ("noncommercial", "by-nc", "-nc", "noderiv", "by-nd", "-nd", "fair use", "non-free", "all rights reserved")):
        return None

    if "cc0" in combined or "creative commons zero" in combined or "publicdomain/zero" in combined:
        return ("CC0 1.0", "CC0", 4)

    if (
        "public domain" in combined
        or "publicdomain/mark" in combined
        or re.search(r"\bpd(?:\b|[-_])", combined)
        or "no restrictions" in combined
    ):
        return ("Public Domain", "Public Domain", 3)

    # Check CC BY-SA before CC BY so BY-SA isn't misclassified as BY
    if "by-sa" in combined or "attribution-sharealike" in combined or "cc by-sa" in combined:
        if not allow_sa:
            return None
        ver_m = re.search(r"(?:by-sa|sharealike)[\s/_\-]*([1-4]\.[05])", combined)
        ver = ver_m.group(1) if ver_m else "4.0"
        return (f"CC-BY-SA {ver}", "CC-BY-SA", 1)

    if "cc by" in combined or "cc-by" in combined or "/licenses/by/" in combined or "creative commons attribution" in combined:
        ver_m = re.search(r"(?:cc[- ]by|/licenses/by/)[\s/_\-]*([1-4]\.[05])", combined)
        ver = ver_m.group(1) if ver_m else "4.0"
        return (f"CC-BY {ver}", "CC-BY", 2)

    return None


class LicenseCleanImageFetcher:
    """Batch-capable Wikipedia + Wikimedia Commons + Openverse fetcher."""

    ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/svg+xml"}

    def __init__(self, *, allow_sa: bool = True, prefer_non_sa: bool = True) -> None:
        self.allow_sa = allow_sa
        self.prefer_non_sa = prefer_non_sa
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _get_json(self, url: str, params: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
        delay = 1.0
        for attempt in range(4):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", delay))
                    time.sleep(min(retry_after, 8.0))
                    delay *= 2.0
                    continue
                if resp.status_code >= 500:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                if resp.status_code != 200:
                    return {}
                return resp.json()
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return {}
                time.sleep(delay)
                delay *= 2.0
        return {}

    def prefetch_wikipedia_leads(self, specs: list[NounSpec]) -> dict[str, dict[str, Any]]:
        """Batch-fetch Wikipedia `pageimage` + intro sentence for up to 40 nouns per HTTP call."""
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(specs), 35):
            chunk = specs[start : start + 35]
            titles = "|".join(s.wiki_title for s in chunk)
            data = self._get_json(WIKI_API, {
                "action": "query",
                "format": "json",
                "redirects": 1,
                "titles": titles,
                "prop": "pageimages|extracts",
                "piprop": "name|thumbnail|original",
                "pithumbsize": 250,
                "exintro": 1,
                "explaintext": 1,
                "exsentences": 1,
            })
            query = data.get("query") or {}
            redirect_map: dict[str, str] = {}
            for norm in query.get("normalized") or []:
                redirect_map[norm["from"]] = norm["to"]
            for red in query.get("redirects") or []:
                redirect_map[red["from"]] = red["to"]

            pages_by_title = {
                p.get("title"): p
                for p in (query.get("pages") or {}).values()
                if isinstance(p, dict) and p.get("title")
            }

            file_titles: list[str] = []
            spec_to_page: list[tuple[NounSpec, dict[str, Any]]] = []
            for s in chunk:
                resolved = s.wiki_title
                visited = set()
                while resolved in redirect_map and resolved not in visited:
                    visited.add(resolved)
                    resolved = redirect_map[resolved]
                page = pages_by_title.get(resolved) or pages_by_title.get(s.wiki_title) or {}
                spec_to_page.append((s, page))
                pimg = page.get("pageimage")
                if pimg:
                    file_titles.append(f"File:{pimg}")

            commons_meta = self._fetch_commons_file_metadata(file_titles) if file_titles else {}
            for s, page in spec_to_page:
                pimg = page.get("pageimage")
                meta = commons_meta.get(f"File:{pimg}".replace("_", " ")) if pimg else None
                out[s.noun] = {
                    "extract": str(page.get("extract") or "").strip(),
                    "lead_candidate": meta,
                }
        return out

    def _fetch_commons_file_metadata(self, file_titles: list[str]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for start in range(0, len(file_titles), 35):
            batch = file_titles[start : start + 35]
            data = self._get_json(COMMONS_API, {
                "action": "query",
                "format": "json",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime|size",
                "iiurlwidth": 250,
            })
            for page in ((data.get("query") or {}).get("pages") or {}).values():
                cand = self._parse_commons_page(page)
                if cand is not None:
                    result[str(page.get("title") or "").replace("_", " ")] = cand
        return result

    def _parse_commons_page(self, page: dict[str, Any]) -> Optional[dict[str, Any]]:
        title = str(page.get("title") or "")
        ii_list = page.get("imageinfo") or []
        if not title or not ii_list:
            return None
        ii = ii_list[0]
        mime = str(ii.get("mime") or "").lower()
        if mime not in self.ALLOWED_MIMES:
            return None
        # Skip multi-page PDF/DjVu/audio/video disguised files
        lower_title = title.lower()
        if any(lower_title.endswith(ext) for ext in (".pdf", ".djvu", ".ogg", ".ogv", ".webm", ".tif", ".tiff")):
            return None

        ext = ii.get("extmetadata") or {}
        short_lic = _clean_html_meta(ext.get("LicenseShortName"))
        usage = _clean_html_meta(ext.get("UsageTerms"))
        lic_url = _clean_html_meta(ext.get("LicenseUrl"))
        nonfree = _clean_html_meta(ext.get("NonFree"))

        lic_check = classify_clean_license(
            short_lic, usage, lic_url, nonfree, allow_sa=self.allow_sa
        )
        if lic_check is None:
            return None

        norm_lic, lic_family, lic_rank = lic_check
        thumb_url = str(ii.get("thumburl") or "")
        orig_url = str(ii.get("url") or "")
        size_bytes = int(ii.get("size") or 0)
        fetch_url = thumb_url or (orig_url if size_bytes <= 2_500_000 else "")
        if not fetch_url:
            return None

        artist = _clean_html_meta(ext.get("Artist")) or _clean_html_meta(ext.get("Credit")) or "Wikimedia Commons contributor"
        if len(artist) > 160:
            artist = artist[:157] + "..."

        desc_url = str(ii.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}")
        return {
            "title": title,
            "mime": mime,
            "fetch_url": fetch_url,
            "original_url": orig_url,
            "source_page_url": desc_url,
            "source_provider": "wikimedia_commons",
            "creator": artist,
            "license": norm_lic,
            "license_family": lic_family,
            "license_rank": lic_rank,
            "license_url": lic_url or self._default_license_url(lic_family),
        }

    @staticmethod
    def _default_license_url(family: str) -> str:
        return {
            "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
            "Public Domain": "https://creativecommons.org/publicdomain/mark/1.0/",
            "CC-BY": "https://creativecommons.org/licenses/by/4.0/",
            "CC-BY-SA": "https://creativecommons.org/licenses/by-sa/4.0/",
        }.get(family, "https://commons.wikimedia.org/wiki/Commons:Licensing")

    def search_commons_candidates(self, spec: NounSpec) -> list[dict[str, Any]]:
        queries = [
            f'"{spec.wiki_title}" filetype:bitmap',
            f"{spec.noun} {spec.category.replace('_', ' ')} filetype:bitmap",
            f"{spec.noun}",
        ]
        candidates: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        noun_tok = spec.noun.lower().split()[0]

        for q in queries:
            data = self._get_json(COMMONS_API, {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": q,
                "gsrnamespace": 6,
                "gsrlimit": 10,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime|size",
                "iiurlwidth": 250,
            })
            pages = ((data.get("query") or {}).get("pages") or {}).values()
            for page in pages:
                cand = self._parse_commons_page(page)
                if cand is None or cand["title"] in seen_titles:
                    continue
                seen_titles.add(cand["title"])
                t_low = cand["title"].lower()
                # Relevance bonus when filename explicitly names the noun or wiki_title
                rel_bonus = 2 if (noun_tok in t_low or spec.wiki_title.lower() in t_low) else 0
                cand["_score"] = cand["license_rank"] * 3 + rel_bonus
                candidates.append(cand)
            if any(c["license_rank"] >= 2 and c.get("_score", 0) >= 8 for c in candidates):
                break

        candidates.sort(key=lambda c: c.get("_score", 0), reverse=True)
        return candidates

    def search_openverse_fallback(self, spec: NounSpec) -> list[dict[str, Any]]:
        lic_param = "cc0,pdm,by,by-sa" if self.allow_sa else "cc0,pdm,by"
        data = self._get_json(OPENVERSE_API, {
            "q": spec.noun,
            "license": lic_param,
            "page_size": 5,
        }, timeout=8.0)
        out: list[dict[str, Any]] = []
        for item in data.get("results") or []:
            lic_code = str(item.get("license") or "").lower()
            lic_ver = str(item.get("license_version") or "4.0")
            lic_check = classify_clean_license(
                f"cc {lic_code} {lic_ver}" if lic_code not in {"pdm", "cc0"} else lic_code,
                "",
                str(item.get("license_url") or ""),
                allow_sa=self.allow_sa,
            )
            if lic_check is None:
                continue
            thumb = str(item.get("thumbnail") or item.get("url") or "")
            if not thumb:
                continue
            norm_lic, lic_family, lic_rank = lic_check
            out.append({
                "title": str(item.get("title") or spec.noun),
                "mime": "image/jpeg",
                "fetch_url": thumb,
                "original_url": str(item.get("url") or thumb),
                "source_page_url": str(item.get("foreign_landing_url") or thumb),
                "source_provider": "openverse",
                "creator": str(item.get("creator") or "Openverse contributor")[:160],
                "license": norm_lic,
                "license_family": lic_family,
                "license_rank": lic_rank,
                "license_url": str(item.get("license_url") or self._default_license_url(lic_family)),
            })
        return out

    def fetch_best_image(
        self,
        spec: NounSpec,
        prefetched: Optional[dict[str, Any]] = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        lead = (prefetched or {}).get("lead_candidate")
        if lead is not None:
            # If lead is already CC0 / PD / CC-BY (rank >= 2), put it #1 immediately.
            # If lead is CC-BY-SA and prefer_non_sa is set, also check Commons for a
            # high-relevance CC0 / PD / CC-BY candidate first, keeping lead as clean backup!
            if lead["license_rank"] >= 2 or not self.prefer_non_sa:
                candidates.append(lead)
            else:
                commons_cands = self.search_commons_candidates(spec)
                non_sa = [c for c in commons_cands if c["license_rank"] >= 2 and c.get("_score", 0) >= 8]
                candidates.extend(non_sa[:2])
                candidates.append(lead)
                candidates.extend(c for c in commons_cands if c not in non_sa[:2])

        if not candidates:
            candidates.extend(self.search_commons_candidates(spec))
        if len(candidates) < 2:
            candidates.extend(self.search_openverse_fallback(spec))

        last_err = "no_license_clean_candidates"
        for cand in candidates[:6]:
            try:
                img = self._download_pil_rgb(cand["fetch_url"])
                if img is not None:
                    return img, cand
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                continue
        raise RuntimeError(f"Could not fetch license-clean image for {spec.noun!r}: {last_err}")

    def _download_pil_rgb(self, url: str) -> Optional[Image.Image]:
        delay = 1.0
        for attempt in range(3):
            resp = self.session.get(url, timeout=15.0)
            if resp.status_code == 429:
                time.sleep(min(float(resp.headers.get("Retry-After", delay)), 6.0))
                delay *= 2.0
                continue
            if resp.status_code != 200 or not resp.content:
                return None
            with Image.open(io.BytesIO(resp.content)) as im:
                # Composite transparency cleanly onto neutral mid-light canvas so silhouettes stay crisp
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    rgba = im.convert("RGBA")
                    bg = Image.new("RGBA", rgba.size, (242, 242, 242, 255))
                    composed = Image.alpha_composite(bg, rgba).convert("RGB")
                else:
                    composed = im.convert("RGB")
                return composed
        return None


# ============================================================================
# 4. C4rr13rX Speech Synthesis Pipeline (16 kHz 16-bit Mono WAV)
# ============================================================================

class SpeechSynthesizerPipeline:
    """Synthesizes clean 16 kHz 16-bit mono PCM WAV spoken audio with C4rr13rX
    pronunciation disambiguation and silence trimming (no upscaling/embellishment).
    """

    SAMPLE_RATE = 16_000

    def __init__(self, engine: str = "auto") -> None:
        self.engine_requested = engine
        self._sapi_voice = None
        self._sapi_format = None
        self.active_engine = self._init_engine(engine)

    def _init_engine(self, engine: str) -> str:
        if engine in ("auto", "sapi") and os.name == "nt":
            try:
                import win32com.client  # type: ignore
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                # Select clear English desktop voice (Microsoft Zira Desktop or David Desktop)
                for v in voice.GetVoices():
                    desc = str(v.GetDescription())
                    if "en-us" in desc.lower() or "english (united states)" in desc.lower() or "david" in desc.lower() or "zira" in desc.lower():
                        voice.Voice = v
                        break
                voice.Rate = 0
                voice.Volume = 100
                fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
                fmt.Type = 18  # SAFT16kHz16BitMono
                self._sapi_voice = voice
                self._sapi_format = fmt
                return "c4rr13rx_disambiguator+windows_sapi_16khz_pcm16"
            except Exception:  # noqa: BLE001
                if engine == "sapi":
                    raise
        raise RuntimeError("No working 16kHz WAV TTS synthesizer available on host")

    def synthesize_to_wav_file(self, text: str, out_wav: Path) -> tuple[np.ndarray, float]:
        """Synthesize spoken English text -> trimmed 16 kHz 16-bit mono WAV at `out_wav`.
        Returns `(pcm_float32_array, duration_seconds)`.
        """
        import win32com.client  # type: ignore

        clean_text = ContextualLinguisticDisambiguator.disambiguate(text, lang="en")
        raw_tmp = out_wav.with_suffix(".raw_sapi.wav")
        try:
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Format = self._sapi_format
            stream.Open(str(raw_tmp), 3, False)
            try:
                self._sapi_voice.AudioOutputStream = stream
                self._sapi_voice.Speak(clean_text)
            finally:
                stream.Close()

            with wave.open(str(raw_tmp), "rb") as wr:
                n_frames = wr.getnframes()
                raw_bytes = wr.readframes(n_frames)
                rate = wr.getframerate()
                channels = wr.getnchannels()
        finally:
            raw_tmp.unlink(missing_ok=True)

        samples_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
        if channels > 1:
            samples_i16 = samples_i16.reshape(-1, channels).mean(axis=1).astype(np.int16)

        # Trim dead leading/trailing silence while keeping a gentle 35ms natural margin
        abs_s = np.abs(samples_i16.astype(np.int32))
        active_idx = np.flatnonzero(abs_s > 220)
        pad = int(0.035 * rate)
        if active_idx.size > 0:
            lo = max(0, int(active_idx[0]) - pad)
            hi = min(samples_i16.size, int(active_idx[-1]) + pad)
            trimmed_i16 = samples_i16[lo:hi]
        else:
            trimmed_i16 = samples_i16[: max(1600, min(samples_i16.size, 8000))]

        with wave.open(str(out_wav), "wb") as ww:
            ww.setnchannels(1)
            ww.setsampwidth(2)
            ww.setframerate(self.SAMPLE_RATE)
            ww.writeframes(trimmed_i16.tobytes())

        pcm_f32 = trimmed_i16.astype(np.float32) / 32768.0
        duration_sec = round(trimmed_i16.size / float(self.SAMPLE_RATE), 3)
        return pcm_f32, duration_sec


# ============================================================================
# 5. Minimal-Neuron Canonical Decomposition (`ImageBitsEncoder` + `AudioBitsEncoder`
#    + Strictly Byte-Disjoint Shared Relational Primitives)
# ============================================================================

# Guard per skeleton_morphology.identity.toml: every shared primitive label in a
# byte-passthrough pool MUST be substring-disjoint so shorter labels are never
# swallowed by longer labels during concept decode!
ALL_SHARED_VISUAL_PRIMITIVES = (
    "vcol:crimson", "vcol:amber", "vcol:gold", "vcol:lime", "vcol:emerald",
    "vcol:cyan", "vcol:cobalt", "vcol:violet", "vcol:magenta",
    "vcol:obsidian", "vcol:ivory", "vcol:slate", "vcol:umber",
    "vshp:orb", "vshp:wide_bar", "vshp:tall_col", "vshp:quad_box", "vshp:complex_form",
    "vsym:bilat_hi", "vsym:asym_org",
    "vedg:calm_surf", "vedg:fine_tex", "vedg:horiz_rib", "vedg:vert_rib", "vedg:diag_arc",
    "vlum:radiant", "vlum:midtone", "vlum:shadowed",
    "vsat:chromatic", "vsat:achromat",
)

ALL_SHARED_AUDIO_PRIMITIVES = (
    "adur:short_1syl", "adur:med_2syl", "adur:long_poly",
    "aspec:treble_sib", "aspec:bassy_voic", "aspec:mid_formant",
    "aenv:crisp_burst", "aenv:steady_tone",
)


def assert_byte_disjoint(tokens: tuple[str, ...]) -> None:
    for i, a in enumerate(tokens):
        assert len(a.encode("utf-8")) <= 24, f"Token exceeds 24-byte concept cap: {a!r}"
        for j, b in enumerate(tokens):
            if i != j and a in b:
                raise AssertionError(f"Substring collision in primitive vocabulary: {a!r} in {b!r}")


assert_byte_disjoint(ALL_SHARED_VISUAL_PRIMITIVES)
assert_byte_disjoint(ALL_SHARED_AUDIO_PRIMITIVES)


def normalize_canonical_image_64(img: Image.Image, out_png: Path) -> np.ndarray:
    """Downsample incoming RGB image cleanly to 64x64 without upscaling embellishment,
    save as compact 64-color indexed PNG (`out_png`), and return `(64, 64, 3)` uint8 RGB.
    """
    w, h = img.size
    # Aspect-preserving fit onto 64x64 neutral canvas; never sharpen or hallucinate detail
    scale = min(64.0 / max(1, w), 64.0 / max(1, h))
    nw = max(1, min(64, int(round(w * scale))))
    nh = max(1, min(64, int(round(h * scale))))
    resample = Image.Resampling.BILINEAR if scale < 1.0 else Image.Resampling.NEAREST
    resized = img.resize((nw, nh), resample=resample)

    # Sample border median color so padding matches natural background
    arr_r = np.asarray(resized, dtype=np.uint8)
    border_px = np.concatenate([
        arr_r[0, :, :], arr_r[-1, :, :], arr_r[:, 0, :], arr_r[:, -1, :]
    ], axis=0)
    bg_rgb = tuple(int(x) for x in np.median(border_px, axis=0))

    canvas = Image.new("RGB", (64, 64), bg_rgb)
    canvas.paste(resized, ((64 - nw) // 2, (64 - nh) // 2))

    # Quantize to 64 adaptive colors -> minimal entropy, ~1.5-3.0 KB PNG, faithful appearance
    quantized = canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized.save(out_png, format="PNG", optimize=True)
    canonical_rgb = np.asarray(quantized.convert("RGB"), dtype=np.uint8)
    return canonical_rgb


def _rgb_to_hsv_bins(r: int, g: int, b: int, hue_bins: int = 16, sat_bins: int = 8, val_bins: int = 8) -> tuple[int, int, int]:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax = max(rf, gf, bf)
    cmin = min(rf, gf, bf)
    delta = cmax - cmin
    if delta < 1e-6:
        hue = 0.0
    elif abs(cmax - rf) < 1e-6:
        hue = 60.0 * (((gf - bf) / delta) % 6.0)
    elif abs(cmax - gf) < 1e-6:
        hue = 60.0 * (((bf - rf) / delta) + 2.0)
    else:
        hue = 60.0 * (((rf - gf) / delta) + 4.0)
    hue = ((hue + 360.0) % 360.0) / 360.0
    sat = 0.0 if cmax < 1e-6 else (delta / cmax)
    val = cmax
    return (
        min(hue_bins - 1, int(hue * hue_bins)),
        min(sat_bins - 1, int(sat * sat_bins)),
        min(val_bins - 1, int(val * val_bins)),
    )


def decompose_visual_minimal_neurons(rgb64: np.ndarray) -> dict[str, Any]:
    """Exact parity with `crates/core/src/streaming/image_bits.rs` (`ImageBitsEncoder`)
    PLUS an 8x8 compact reconstructible cell grid and byte-disjoint shared primitives.
    """
    h, w, _ = rgb64.shape  # 64 x 64
    grey = (
        0.299 * rgb64[:, :, 0].astype(np.float32)
        + 0.587 * rgb64[:, :, 1].astype(np.float32)
        + 0.114 * rgb64[:, :, 2].astype(np.float32)
    )

    grid_x, grid_y = 8, 8
    stride_x, stride_y = 8, 8
    salience_threshold = 20
    sample_every = 4  # sample_rate = 0.25

    labels_set: set[str] = set()
    active_pixels = 0
    counter = 0
    edge_counts = {"H": 0, "V": 0, "D1": 0, "D2": 0}

    for y in range(h):
        for x in range(w):
            if int(grey[y, x]) < salience_threshold:
                continue
            counter += 1
            if counter % sample_every != 0:
                continue
            active_pixels += 1
            zx = min(grid_x - 1, x // stride_x)
            zy = min(grid_y - 1, y // stride_y)
            r, g, b = int(rgb64[y, x, 0]), int(rgb64[y, x, 1]), int(rgb64[y, x, 2])
            hb, sb, vb = _rgb_to_hsv_bins(r, g, b, 16, 8, 8)
            labels_set.add(f"img:z{zx}_{zy}")
            labels_set.add(f"img:h{hb}")
            labels_set.add(f"img:s{sb}")
            labels_set.add(f"img:v{vb}")
            labels_set.add(f"img:h{hb}_z{zx}_{zy}")

            if 0 < x < w - 1 and 0 < y < h - 1:
                gx = (
                    -grey[y - 1, x - 1] + grey[y - 1, x + 1]
                    - 2.0 * grey[y, x - 1] + 2.0 * grey[y, x + 1]
                    - grey[y + 1, x - 1] + grey[y + 1, x + 1]
                )
                gy = (
                    -grey[y - 1, x - 1] - 2.0 * grey[y - 1, x] - grey[y - 1, x + 1]
                    + grey[y + 1, x - 1] + 2.0 * grey[y + 1, x] + grey[y + 1, x + 1]
                )
                mag = math.sqrt(float(gx * gx + gy * gy))
                if mag > 30.0:
                    ang = abs(math.degrees(math.atan2(float(gy), float(gx)))) % 180.0
                    if ang < 22.5 or ang >= 157.5:
                        ek = "H"
                    elif ang < 67.5:
                        ek = "D1"
                    elif ang < 112.5:
                        ek = "V"
                    else:
                        ek = "D2"
                    edge_counts[ek] += 1
                    labels_set.add(f"img:edge{ek}_z{zx}_{zy}")

    # Compact 8x8 cell grid (64 cells -> 2 hex chars per cell = 128 chars total):
    # high nibble = hue16 (0..f), low nibble = (sat2 << 2) | val2
    cell_hex_parts: list[str] = []
    fg_hue_votes = np.zeros(16, dtype=np.int32)
    fg_sats: list[int] = []
    fg_vals: list[int] = []

    border_lums = np.concatenate([grey[0, :], grey[-1, :], grey[:, 0], grey[:, -1]])
    bg_lum = float(np.median(border_lums))
    fg_mask = np.abs(grey - bg_lum) > 22.0
    if int(fg_mask.sum()) < 64:
        # Fallback central foreground window when object fills frame
        fg_mask = np.zeros((64, 64), dtype=bool)
        fg_mask[10:54, 10:54] = True

    for cy in range(8):
        for cx in range(8):
            patch = rgb64[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8]
            mr, mg, mb = (int(v) for v in patch.reshape(-1, 3).mean(axis=0))
            hb, sb, vb = _rgb_to_hsv_bins(mr, mg, mb, 16, 8, 8)
            packed_byte = ((hb & 0xF) << 4) | (((sb >> 1) & 0x3) << 2) | ((vb >> 1) & 0x3)
            cell_hex_parts.append(f"{packed_byte:02x}")
            if fg_mask[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8].mean() > 0.25:
                if sb >= 2:
                    fg_hue_votes[hb] += 2
                fg_sats.append(sb)
                fg_vals.append(vb)

    # Derive shared cross-noun visual primitives (strictly byte-disjoint!)
    shared: list[str] = []
    mean_sat = float(np.mean(fg_sats)) if fg_sats else 0.0
    mean_val = float(np.mean(fg_vals)) if fg_vals else float(grey.mean() / 32.0)

    if mean_sat < 1.6:
        if mean_val <= 2.2:
            shared.append("vcol:obsidian")
        elif mean_val >= 6.0:
            shared.append("vcol:ivory")
        else:
            shared.append("vcol:slate")
        shared.append("vsat:achromat")
    else:
        dom_h = int(np.argmax(fg_hue_votes))
        if mean_val < 3.6 and dom_h in (1, 2):
            shared.append("vcol:umber")
        else:
            hue_map = {
                0: "vcol:crimson", 1: "vcol:amber", 2: "vcol:gold", 3: "vcol:lime",
                4: "vcol:emerald", 5: "vcol:emerald", 6: "vcol:emerald",
                7: "vcol:cyan", 8: "vcol:cyan", 9: "vcol:cobalt", 10: "vcol:cobalt",
                11: "vcol:violet", 12: "vcol:violet", 13: "vcol:magenta",
                14: "vcol:magenta", 15: "vcol:crimson",
            }
            shared.append(hue_map[dom_h])
        shared.append("vsat:chromatic")

    # Foreground bounding box & shape compactness
    ys, xs = np.nonzero(fg_mask)
    bw = max(1, int(xs.max() - xs.min() + 1))
    bh = max(1, int(ys.max() - ys.min() + 1))
    aspect = bw / float(bh)
    fill_ratio = float(fg_mask.sum()) / float(bw * bh)
    # Bilateral left-right symmetry score
    sym_diff = float(np.abs(grey - grey[:, ::-1]).mean())

    if aspect > 1.42:
        shared.append("vshp:wide_bar")
    elif aspect < 0.70:
        shared.append("vshp:tall_col")
    elif 0.62 <= fill_ratio <= 0.84 and sym_diff < 34.0:
        shared.append("vshp:orb")
    elif fill_ratio > 0.84:
        shared.append("vshp:quad_box")
    else:
        shared.append("vshp:complex_form")

    shared.append("vsym:bilat_hi" if sym_diff < 26.0 else "vsym:asym_org")

    total_edges = sum(edge_counts.values())
    if total_edges < 18:
        shared.append("vedg:calm_surf")
    elif total_edges > 95:
        shared.append("vedg:fine_tex")
    else:
        hv = edge_counts["H"]
        vv = edge_counts["V"]
        dv = edge_counts["D1"] + edge_counts["D2"]
        if dv >= hv and dv >= vv:
            shared.append("vedg:diag_arc")
        elif hv >= vv:
            shared.append("vedg:horiz_rib")
        else:
            shared.append("vedg:vert_rib")

    if mean_val >= 5.5:
        shared.append("vlum:radiant")
    elif mean_val <= 2.5:
        shared.append("vlum:shadowed")
    else:
        shared.append("vlum:midtone")

    sorted_labels = sorted(labels_set)
    cell_grid_hex = "".join(cell_hex_parts)
    # Compact brain frame (< 220 bytes): shared primitives + active structural zone anchors
    zone_anchors = [l for l in sorted_labels if l.startswith("img:edge") or l.startswith("img:h") and "_z" not in l][:12]
    compact_brain_tokens = shared + zone_anchors

    return {
        "canonical_resolution": "64x64",
        "cell_grid_8x8_hex": cell_grid_hex,
        "shared_primitives": shared,
        "compact_brain_tokens": compact_brain_tokens,
        "image_bits_labels": sorted_labels,
        "active_pixels": active_pixels,
    }


def decompose_audio_minimal_neurons(pcm_f32: np.ndarray, noun_text: str) -> dict[str, Any]:
    """Exact parity with `crates/core/src/streaming/audio_bits.rs` (`AudioBitsEncoder`)
    PLUS an 8x8 temporal-spectral nibble grid and byte-disjoint shared phonetic primitives.
    """
    ws = 512
    hop = 256
    freq_bins = 32
    amp_bins = 8
    time_slots = 16
    silence_threshold = 0.005

    n = int(pcm_f32.size)
    total_windows = ((n - ws) // hop + 1) if n >= ws else 0
    hann = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(ws, dtype=np.float32) / (ws - 1)))).astype(np.float32)

    labels_set: set[str] = set()
    active_windows = 0
    grid_8x8 = np.zeros((8, 8), dtype=np.float32)
    rms_track: list[float] = []
    high_energy = 0.0
    low_energy = 0.0

    for win_idx in range(total_windows):
        start = win_idx * hop
        window = pcm_f32[start : start + ws]
        rms = float(np.sqrt(np.mean(window * window)))
        if rms < silence_threshold:
            continue
        active_windows += 1
        rms_track.append(rms)

        t_slot = min(time_slots - 1, int((win_idx / max(1, total_windows)) * time_slots))
        t8 = min(7, int((win_idx / max(1, total_windows)) * 8))

        spectrum = np.abs(np.fft.rfft(window * hann))  # length ws//2 + 1 = 257
        nyq = int(spectrum.size)

        for band in range(freq_bins):
            lo = (band * nyq) // freq_bins
            hi = min(nyq, ((band + 1) * nyq) // freq_bins)
            if lo >= hi:
                continue
            energy = float(np.mean(spectrum[lo:hi]))
            norm = min(1.0, energy / (ws * 0.5))
            b8 = band // 4
            grid_8x8[t8, b8] = max(grid_8x8[t8, b8], norm)
            if band < 8:
                low_energy += energy
            elif band >= 16:
                high_energy += energy

            if norm < 0.005:
                continue
            amp_bin = min(amp_bins - 1, int(norm * amp_bins))
            labels_set.add(f"aud:freq{band}")
            labels_set.add(f"aud:amp{amp_bin}")
            labels_set.add(f"aud:freq{band}_t{t_slot}")
            labels_set.add(f"aud:amp{amp_bin}_t{t_slot}")

        loud_bin = min(amp_bins - 1, int(rms * amp_bins))
        labels_set.add(f"aud:loud{loud_bin}")
        labels_set.add(f"aud:t{t_slot}")

    # Quantize 8x8 temporal-spectral envelope to 0..f hex nibbles (64 chars)
    gmax = float(grid_8x8.max()) if grid_8x8.max() > 1e-6 else 1.0
    nibble_grid = np.clip(np.round((grid_8x8 / gmax) * 15.0), 0, 15).astype(np.uint8)
    spectral_hex = "".join(f"{int(v):x}" for v in nibble_grid.flatten())

    # Shared phonetic/acoustic primitives (strictly byte-disjoint!)
    clean_w = re.sub(r"[^a-z]", "", noun_text.lower())
    vowel_groups = re.findall(r"[aeiouy]+", clean_w)
    est_syl = max(1, len(vowel_groups) - (1 if clean_w.endswith("e") and len(vowel_groups) > 1 else 0))

    shared: list[str] = []
    if est_syl == 1:
        shared.append("adur:short_1syl")
    elif est_syl == 2:
        shared.append("adur:med_2syl")
    else:
        shared.append("adur:long_poly")

    if high_energy > low_energy * 0.55:
        shared.append("aspec:treble_sib")
    elif low_energy > high_energy * 3.2:
        shared.append("aspec:bassy_voic")
    else:
        shared.append("aspec:mid_formant")

    if len(rms_track) >= 3 and (max(rms_track) / max(1e-4, float(np.mean(rms_track)))) > 1.75:
        shared.append("aenv:crisp_burst")
    else:
        shared.append("aenv:steady_tone")

    # Byte-disjoint onset & nucleus anchors (fixed 2-char hex/code suffix so never substrings!)
    onset_ch = clean_w[0] if clean_w else "x"
    coda_ch = clean_w[-1] if clean_w else "x"
    first_vow = vowel_groups[0][0] if vowel_groups else "a"
    shared.append(f"aons:{onset_ch}_0")
    shared.append(f"avow:{first_vow}_1")
    shared.append(f"acod:{coda_ch}_2")

    sorted_labels = sorted(labels_set)
    freq_anchors = [l for l in sorted_labels if l.startswith("aud:freq") and "_t" not in l][:10]
    compact_brain_tokens = shared + freq_anchors

    return {
        "sample_rate_hz": 16_000,
        "spectral_grid_8x8_hex": spectral_hex,
        "shared_primitives": shared,
        "compact_brain_tokens": compact_brain_tokens,
        "audio_bits_labels": sorted_labels,
        "windows_processed": total_windows,
        "active_windows": active_windows,
    }


def slugify_noun(noun: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", noun.strip().lower()).strip("_")


# ============================================================================
# 6. End-to-End Curation Loop with Cloud Verify-and-Delete
# ============================================================================

def load_existing_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not manifest_path.exists():
        return existing
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("noun") and rec.get("cloud_verified"):
                    existing[str(rec["noun"]).lower()] = rec
            except Exception:  # noqa: BLE001
                continue
    return existing


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Curate license-clean noun -> canonical image + spoken audio + minimal-neuron dictionary."
    )
    ap.add_argument("--limit", type=int, default=90, help="Max new nouns to curate in this run (default: 90).")
    ap.add_argument("--max-batch-size", type=int, default=16, help="Max elastic prefetch batch when RAM >= 14.5 GB.")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip nouns already verified in manifest.")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Rebuild manifest from scratch.")
    ap.add_argument("--disallow-sa", action="store_true", help="Restrict strictly to CC0, Public Domain, and CC-BY only.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cloud-prefix", default="noun_dictionary")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "noun_multimodal_dictionary.jsonl"
    summary_path = out_dir / "noun_multimodal_summary.json"

    governor = PoliteResourceGovernor(max_batch_size=args.max_batch_size)
    init_mem = governor.sample()

    cloud = MultimodalCloudSync(base_prefix=args.cloud_prefix)
    fetcher = LicenseCleanImageFetcher(allow_sa=not args.disallow_sa, prefer_non_sa=True)
    tts = SpeechSynthesizerPipeline(engine="auto")

    catalog = build_noun_catalog()
    existing = load_existing_manifest(manifest_path) if args.resume else {}
    if not args.resume and manifest_path.exists():
        manifest_path.unlink()

    pending = [s for s in catalog if s.noun.lower() not in existing]
    target_specs = pending[: max(0, int(args.limit))]

    print(
        f"[curate] priority={governor.priority_applied} | avail_ram={init_mem['avail_gb']} GB "
        f"(floor={governor.EFFECTIVE_FLOOR_GB} GB, scale_up={governor.SCALE_UP_RAM_GB} GB) | "
        f"bucket={cloud.provider}://{cloud.bucket}/{cloud.base_prefix}",
        flush=True,
    )
    print(
        f"[curate] catalog={len(catalog)} concrete nouns | existing={len(existing)} | "
        f"curating_target={len(target_specs)}",
        flush=True,
    )

    curated_this_run = 0
    skipped_failures: list[dict[str, str]] = []
    license_counts: dict[str, int] = {}
    bytes_uploaded = 0
    cloud_objects_verified = 0
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="w1z4rd_noun_curate_", dir=str(out_dir)) as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        idx = 0
        # Allow slightly wider candidate pool so if 1-2 esoteric images fail license check we still hit `--limit`
        candidate_pool = pending[: min(len(pending), int(args.limit) + 20)]

        while idx < len(candidate_pool) and curated_this_run < int(args.limit):
            mem_state = governor.sample()
            elastic_batch = min(
                int(mem_state["batch_size"]),
                int(args.limit) - curated_this_run,
                len(candidate_pool) - idx,
            )
            batch_specs = candidate_pool[idx : idx + max(1, elastic_batch)]
            idx += len(batch_specs)

            prefetched_map = fetcher.prefetch_wikipedia_leads(batch_specs)

            for spec in batch_specs:
                if curated_this_run >= int(args.limit):
                    break

                slug = slugify_noun(spec.noun)
                tmp_png = tmp_root / f"{slug}.png"
                tmp_word_wav = tmp_root / f"{slug}.wav"
                tmp_sent_wav = tmp_root / f"{slug}_sentence.wav"

                try:
                    pre = prefetched_map.get(spec.noun) or {}
                    pil_img, provenance = fetcher.fetch_best_image(spec, prefetched=pre)

                    # 1. Canonical 64x64 image + minimal-neuron visual decomposition
                    rgb64 = normalize_canonical_image_64(pil_img, tmp_png)
                    del pil_img
                    vis_decomp = decompose_visual_minimal_neurons(rgb64)
                    del rgb64

                    # 2. Spoken English audio (isolated noun + grounding sentence) via C4rr13rX TTS rules
                    grounding_text = spec.grounding_sentence
                    if not grounding_text and pre.get("extract"):
                        grounding_text = str(pre["extract"]).split(". ")[0].strip() + "."

                    pcm_word, word_dur_s = tts.synthesize_to_wav_file(spec.noun, tmp_word_wav)
                    _, sent_dur_s = tts.synthesize_to_wav_file(grounding_text, tmp_sent_wav)
                    aud_decomp = decompose_audio_minimal_neurons(pcm_word, spec.noun)
                    del pcm_word

                    # 3. Cloud Upload -> Remote HEAD/MD5/SHA256 Verify -> Immediate Local unlink()
                    common_meta = {
                        "noun": spec.noun,
                        "license": provenance["license"],
                        "source_provider": provenance["source_provider"],
                    }
                    img_receipt = cloud.upload_verify_and_delete(
                        tmp_png,
                        f"images/{slug}.png",
                        content_type="image/png",
                        metadata=common_meta,
                    )
                    word_aud_receipt = cloud.upload_verify_and_delete(
                        tmp_word_wav,
                        f"audio/{slug}.wav",
                        content_type="audio/wav",
                        metadata=common_meta,
                    )
                    sent_aud_receipt = cloud.upload_verify_and_delete(
                        tmp_sent_wav,
                        f"audio/{slug}_sentence.wav",
                        content_type="audio/wav",
                        metadata=common_meta,
                    )

                    # Strict invariant: zero local temp files remain
                    assert not tmp_png.exists() and not tmp_word_wav.exists() and not tmp_sent_wav.exists()

                    social_primitives = [
                        f"sdom:{spec.domain}",
                        f"scat:{spec.category}",
                        f"saff:{spec.affordance}",
                    ]

                    vis_frame_str = "|".join(vis_decomp["compact_brain_tokens"])
                    aud_frame_str = "|".join(aud_decomp["compact_brain_tokens"])
                    soc_frame_str = "|".join(social_primitives)

                    neuron_budget = {
                        "text_char_atoms": len(spec.noun),
                        "visual_shared_primitives": len(vis_decomp["shared_primitives"]),
                        "visual_compact_tokens": len(vis_decomp["compact_brain_tokens"]),
                        "image_bits_labels": len(vis_decomp["image_bits_labels"]),
                        "audio_shared_primitives": len(aud_decomp["shared_primitives"]),
                        "audio_compact_tokens": len(aud_decomp["compact_brain_tokens"]),
                        "audio_bits_labels": len(aud_decomp["audio_bits_labels"]),
                        "social_primitives": len(social_primitives),
                        "total_minimal_concept_tokens": (
                            1
                            + len(vis_decomp["compact_brain_tokens"])
                            + len(aud_decomp["compact_brain_tokens"])
                            + len(social_primitives)
                        ),
                        "total_co_fire_bytes": (
                            len(spec.noun.encode("utf-8"))
                            + len(vis_frame_str.encode("utf-8"))
                            + len(aud_frame_str.encode("utf-8"))
                            + len(soc_frame_str.encode("utf-8"))
                        ),
                    }

                    record = {
                        "noun": spec.noun,
                        "slug": slug,
                        "domain": spec.domain,
                        "category": spec.category,
                        "affordance": spec.affordance,
                        "corpus_source": spec.corpus_source,
                        "grounding_sentence": grounding_text,
                        "provenance": {
                            "license": provenance["license"],
                            "license_family": provenance["license_family"],
                            "license_url": provenance["license_url"],
                            "creator": provenance["creator"],
                            "source_provider": provenance["source_provider"],
                            "source_page_url": provenance["source_page_url"],
                            "source_image_url": provenance["original_url"] or provenance["fetch_url"],
                            "commons_title": provenance["title"],
                            "tts_pipeline": tts.active_engine,
                        },
                        "cloud_media": {
                            "image_uri": img_receipt.cloud_uri,
                            "image_bytes": img_receipt.byte_size,
                            "image_sha256": img_receipt.sha256,
                            "image_generation": img_receipt.generation_or_etag,
                            "audio_word_uri": word_aud_receipt.cloud_uri,
                            "audio_word_bytes": word_aud_receipt.byte_size,
                            "audio_word_sha256": word_aud_receipt.sha256,
                            "audio_word_duration_s": word_dur_s,
                            "audio_sentence_uri": sent_aud_receipt.cloud_uri,
                            "audio_sentence_bytes": sent_aud_receipt.byte_size,
                            "audio_sentence_sha256": sent_aud_receipt.sha256,
                            "audio_sentence_duration_s": sent_dur_s,
                        },
                        "cloud_verified": bool(
                            img_receipt.verified_in_cloud
                            and word_aud_receipt.verified_in_cloud
                            and sent_aud_receipt.verified_in_cloud
                        ),
                        "local_temp_deleted": bool(
                            img_receipt.deleted_locally
                            and word_aud_receipt.deleted_locally
                            and sent_aud_receipt.deleted_locally
                        ),
                        "visual_decomposition": vis_decomp,
                        "audio_decomposition": aud_decomp,
                        "social_primitives": social_primitives,
                        "neuron_budget": neuron_budget,
                    }

                    with manifest_path.open("a", encoding="utf-8") as mfh:
                        mfh.write(json.dumps(record, ensure_ascii=False) + "\n")

                    curated_this_run += 1
                    lic_fam = provenance["license_family"]
                    license_counts[lic_fam] = license_counts.get(lic_fam, 0) + 1
                    bytes_uploaded += (
                        img_receipt.byte_size + word_aud_receipt.byte_size + sent_aud_receipt.byte_size
                    )
                    cloud_objects_verified += 3

                    if curated_this_run % 10 == 0 or curated_this_run == 1:
                        cur_mem = governor.sample()
                        print(
                            f"  [{curated_this_run}/{args.limit}] {spec.noun:<13} "
                            f"lic={provenance['license']:<12} "
                            f"atoms={neuron_budget['total_minimal_concept_tokens']:<3} "
                            f"img={img_receipt.byte_size}B aud={word_aud_receipt.byte_size}B "
                            f"avail_ram={cur_mem['avail_gb']}GB ({cur_mem['mode']})",
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    skipped_failures.append({"noun": spec.noun, "error": str(exc)[:220]})
                finally:
                    for p in (tmp_png, tmp_word_wav, tmp_sent_wav):
                        p.unlink(missing_ok=True)

                time.sleep(float(mem_state["pace_sleep"]))

            governor.sweep_memory()

    # Final check: zero leftover local media files anywhere in output directory
    leftover_media = MultimodalCloudSync.count_local_media_files(out_dir)
    all_records = load_existing_manifest(manifest_path)

    all_lic_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    avg_tokens = 0.0
    for rec in all_records.values():
        lf = rec.get("provenance", {}).get("license_family", "unknown")
        all_lic_counts[lf] = all_lic_counts.get(lf, 0) + 1
        dm = rec.get("domain", "unknown")
        domain_counts[dm] = domain_counts.get(dm, 0) + 1
        avg_tokens += float(rec.get("neuron_budget", {}).get("total_minimal_concept_tokens", 0))
    if all_records:
        avg_tokens = round(avg_tokens / len(all_records), 2)

    # Mirror mutable JSONL manifest checkpoint to cloud storage and verify updated size/hash
    cloud_manifest_uri = ""
    if manifest_path.exists():
        man_bytes = manifest_path.read_bytes()
        man_sha256, man_md5_b64 = cloud.compute_digests(man_bytes)
        man_key = cloud.full_key("manifests/noun_multimodal_dictionary.jsonl")
        if cloud.provider == "gcs":
            man_blob = cloud.gcs_client.bucket(cloud.bucket).blob(man_key)
            man_blob.metadata = {"sha256": man_sha256}
            man_blob.upload_from_string(
                man_bytes, content_type="application/x-ndjson", checksum="md5", timeout=60
            )
        else:
            cloud.s3_client.put_object(
                Bucket=cloud.bucket,
                Key=man_key,
                Body=man_bytes,
                ContentType="application/x-ndjson",
                Metadata={"sha256": man_sha256},
            )
        man_check = cloud.verify_remote_object(
            cloud.uri_for_key(man_key),
            expected_size=len(man_bytes),
            expected_md5_b64=man_md5_b64,
            expected_sha256=man_sha256,
        )
        if not man_check.get("verified"):
            raise RuntimeError(f"Manifest cloud verification failed: {man_check}")
        cloud_manifest_uri = man_check["cloud_uri"]

    summary_payload = {
        "generated_at_epoch": round(time.time(), 2),
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
        "total_catalog_concrete_nouns": len(catalog),
        "total_verified_nouns_in_manifest": len(all_records),
        "curated_this_run": curated_this_run,
        "cloud_objects_verified_this_run": cloud_objects_verified,
        "cloud_bytes_uploaded_this_run": bytes_uploaded,
        "leftover_local_media_files": leftover_media,
        "cloud_bucket": f"{cloud.provider}://{cloud.bucket}",
        "cloud_manifest_uri": cloud_manifest_uri,
        "license_family_distribution": all_lic_counts,
        "domain_distribution": domain_counts,
        "mean_minimal_concept_tokens_per_noun": avg_tokens,
        "resource_governor": governor.summary(),
        "skipped_candidates": skipped_failures[:15],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    cloud.upload_json_checkpoint("manifests/noun_multimodal_summary.json", summary_payload)

    print("\n[curate] COMPLETE:", json.dumps(summary_payload, indent=2), flush=True)
    if leftover_media != 0:
        raise RuntimeError(f"Expected 0 leftover local media files, found {leftover_media}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
