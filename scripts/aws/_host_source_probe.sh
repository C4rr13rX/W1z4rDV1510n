echo "--- brain_api.rs on host ---"
f=/srv/wizard/project/crates/node/src/brain_api.rs
ls -l "$f" 2>/dev/null
echo "--- does the descending fix exist in the host source? ---"
grep -n 'rev().find_map\|(2..=maximum)' "$f" 2>/dev/null | head -5
echo "--- requested_manifest_component_count ---"
grep -n 'fn requested_manifest_component_count' -A 40 "$f" 2>/dev/null | head -60
