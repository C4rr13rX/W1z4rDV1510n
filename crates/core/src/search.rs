use crate::hardware::BitOp;
use crate::schema::{DynamicState, EnvironmentSnapshot, Position, Symbol, SymbolType};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub cell_size: f64,
    #[serde(default)]
    pub teleport_on_no_path: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            teleport_on_no_path: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathFailureReason {
    MissingTarget,
    StartBlocked,
    GoalBlocked,
    OutOfBounds,
    NoPath,
}

#[derive(Debug, Clone, Default)]
pub struct PathDiagnostics {
    pub cost: f64,
    pub length: f64,
    pub visited_nodes: usize,
    pub constraint_violations: usize,
    pub reached_goal: bool,
}

impl PathDiagnostics {
    fn add_violation(&mut self) {
        self.constraint_violations += 1;
    }

    fn missing_target() -> Self {
        let mut diag = Self::default();
        diag.add_violation();
        diag
    }
}

#[derive(Debug, Clone)]
pub enum PathResult {
    Feasible {
        waypoints: Vec<Position>,
        diagnostics: PathDiagnostics,
    },
    Infeasible {
        reason: PathFailureReason,
        diagnostics: PathDiagnostics,
    },
}

pub struct SearchModule {
    pub config: SearchConfig,
    cached: Mutex<Option<CachedGrid>>,
}

impl SearchModule {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            config,
            cached: Mutex::new(None),
        }
    }

    /// Explicitly clears any cached occupancy grid. Use when external callers know
    /// the environment bounds or symbol metadata changed in ways the signature
    /// might not capture, or before reusing the search module across scenarios.
    pub fn invalidate_cache(&self) {
        let mut guard = self.cached.lock();
        *guard = None;
        debug!(target: "w1z4rdv1510n::search", "invalidated occupancy-grid cache");
    }

    pub fn compute_paths(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        target_state: &DynamicState,
    ) -> HashMap<String, PathResult> {
        let mut results = HashMap::new();
        let grid = self.grid_for_snapshot(snapshot_0);
        for symbol in &snapshot_0.symbols {
            let result = if let Some(target) = target_state.symbol_states.get(&symbol.id) {
                let outcome = self.plan_between(&grid, &symbol.position, &target.position);
                if outcome.reached_goal {
                    PathResult::Feasible {
                        waypoints: outcome.waypoints,
                        diagnostics: outcome.diagnostics,
                    }
                } else {
                    PathResult::Infeasible {
                        reason: outcome.failure_reason.unwrap_or(PathFailureReason::NoPath),
                        diagnostics: outcome.diagnostics,
                    }
                }
            } else {
                PathResult::Infeasible {
                    reason: PathFailureReason::MissingTarget,
                    diagnostics: PathDiagnostics::missing_target(),
                }
            };
            results.insert(symbol.id.clone(), result);
        }
        results
    }

    pub fn enforce_hard_constraints(
        &self,
        snapshot_0: &EnvironmentSnapshot,
        state: &mut DynamicState,
        hardware: Option<&dyn crate::hardware::HardwareBackend>,
    ) {
        if state.symbol_states.is_empty() {
            return;
        }
        let depth_cap = snapshot_0
            .bounds
            .get("depth")
            .copied()
            .unwrap_or(f64::INFINITY);
        let mut lookup = HashMap::new();
        let mut radii = HashMap::new();
        for symbol in &snapshot_0.symbols {
            lookup.insert(symbol.id.clone(), symbol.position);
            radii.insert(symbol.id.clone(), symbol_radius(symbol));
        }
        let grid = self.grid_for_snapshot(snapshot_0);
        let mut occupancy_mask = vec![0u8; grid.blocked.len()];
        let mut cell_assignments: HashMap<usize, Vec<String>> = HashMap::new();
        for (symbol_id, symbol_state) in state.symbol_states.iter_mut() {
            let target_start = match lookup.get(symbol_id) {
                Some(pos) => pos,
                None => {
                    symbol_state.position =
                        clamp_with_depth(&grid.clamp_position(&symbol_state.position), depth_cap);
                    continue;
                }
            };
            let mut desired = grid.clamp_position(&symbol_state.position);
            desired.z = desired.z.clamp(0.0, depth_cap);
            let outcome = self.plan_between(&grid, target_start, &desired);
            if let Some(last) = outcome.waypoints.last() {
                symbol_state.position = clamp_with_depth(last, depth_cap);
            } else if self.config.teleport_on_no_path {
                if let Some(cell) = grid
                    .position_to_cell(&desired)
                    .or_else(|| grid.position_to_cell(target_start))
                    .and_then(|cell| grid.nearest_free_cell(cell))
                {
                    let teleported = grid.cell_center(cell, desired.z);
                    symbol_state.position = clamp_with_depth(&teleported, depth_cap);
                } else {
                    symbol_state.position = clamp_with_depth(&desired, depth_cap);
                }
            } else {
                symbol_state.position = clamp_with_depth(&desired, depth_cap);
            }
            if let Some(cell) = grid.position_to_cell(&symbol_state.position) {
                if let Some(idx) = grid.index(cell) {
                    occupancy_mask[idx] = 1;
                    cell_assignments
                        .entry(idx)
                        .or_default()
                        .push(symbol_id.clone());
                }
            }
        }
        if let Some(hw) = hardware {
            if !grid.blocked.is_empty() {
                let mut static_mask = grid.blocked.clone();
                let mut dynamic_mask = occupancy_mask.clone();
                let mut collision_mask = vec![0u8; static_mask.len()];
                if hw
                    .bulk_bitop(
                        BitOp::And,
                        &mut [
                            static_mask.as_mut_slice(),
                            dynamic_mask.as_mut_slice(),
                            collision_mask.as_mut_slice(),
                        ],
                    )
                    .is_ok()
                {
                    for (idx, value) in collision_mask.iter().enumerate() {
                        if *value == 0 {
                            continue;
                        }
                        let Some(symbols) = cell_assignments.get(&idx) else {
                            continue;
                        };
                        if let Some(cell) = grid.cell_from_index(idx) {
                            if let Some(nearest) = grid.nearest_free_cell(cell) {
                                let repair = grid.cell_center(nearest, 0.0);
                                for symbol_id in symbols {
                                    if let Some(symbol_state) =
                                        state.symbol_states.get_mut(symbol_id)
                                    {
                                        let mut adjusted = repair;
                                        adjusted.z = symbol_state.position.z;
                                        symbol_state.position =
                                            clamp_with_depth(&adjusted, depth_cap);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let ids: Vec<String> = state.symbol_states.keys().cloned().collect();
        for idx in 0..ids.len() {
            for jdx in (idx + 1)..ids.len() {
                let first_id = &ids[idx];
                let second_id = &ids[jdx];
                let Some(first_state) = state.symbol_states.get(first_id) else {
                    continue;
                };
                let Some(second_state) = state.symbol_states.get(second_id) else {
                    continue;
                };
                let r_first = *radii.get(first_id).unwrap_or(&0.5);
                let r_second = *radii.get(second_id).unwrap_or(&0.5);
                let min_dist = r_first + r_second;
                let dx = second_state.position.x - first_state.position.x;
                let dy = second_state.position.y - first_state.position.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist >= min_dist {
                    continue;
                }
                let overlap = min_dist - dist;
                let nx = if dist > 0.0 { dx / dist } else { 1.0 };
                let ny = if dist > 0.0 { dy / dist } else { 0.0 };
                let adjust = overlap / 2.0;
                if let Some(first_mut) = state.symbol_states.get_mut(first_id) {
                    first_mut.position.x -= nx * adjust;
                    first_mut.position.y -= ny * adjust;
                    first_mut.position =
                        clamp_with_depth(&grid.clamp_position(&first_mut.position), depth_cap);
                }
                if let Some(second_mut) = state.symbol_states.get_mut(second_id) {
                    second_mut.position.x += nx * adjust;
                    second_mut.position.y += ny * adjust;
                    second_mut.position =
                        clamp_with_depth(&grid.clamp_position(&second_mut.position), depth_cap);
                }
            }
        }
    }

    fn plan_between(
        &self,
        grid: &OccupancyGrid,
        start: &Position,
        goal: &Position,
    ) -> PathSearchOutcome {
        let mut diagnostics = PathDiagnostics::default();
        let mut start_point = grid.clamp_position(start);
        if !approx_equal(&start_point, start) {
            diagnostics.add_violation();
        }
        let mut goal_point = grid.clamp_position(goal);
        if !approx_equal(&goal_point, goal) {
            diagnostics.add_violation();
        }

        let mut start_cell = match grid.position_to_cell(&start_point) {
            Some(cell) => cell,
            None => {
                diagnostics.add_violation();
                GridCell { x: 0, y: 0 }
            }
        };
        let mut goal_cell = match grid.position_to_cell(&goal_point) {
            Some(cell) => cell,
            None => {
                diagnostics.add_violation();
                GridCell {
                    x: (grid.width_cells as i32).saturating_sub(1),
                    y: (grid.height_cells as i32).saturating_sub(1),
                }
            }
        };

        if grid.is_blocked(start_cell) {
            diagnostics.add_violation();
            if let Some(free) = grid.nearest_free_cell(start_cell) {
                start_cell = free;
                start_point = grid.cell_center(start_cell, start_point.z);
            } else {
                return PathSearchOutcome::blocked(
                    PathFailureReason::StartBlocked,
                    start_point,
                    diagnostics,
                );
            }
        }

        if grid.is_blocked(goal_cell) {
            diagnostics.add_violation();
            if let Some(free) = grid.nearest_free_cell(goal_cell) {
                goal_cell = free;
                goal_point = grid.cell_center(goal_cell, goal_point.z);
            } else {
                return PathSearchOutcome::blocked(
                    PathFailureReason::GoalBlocked,
                    start_point,
                    diagnostics,
                );
            }
        }

        let search = run_a_star(grid, start_cell, goal_cell);
        let mut waypoints = grid.cells_to_waypoints(
            &search.path_cells,
            &start_point,
            &goal_point,
            search.reached_goal,
        );
        if waypoints.is_empty() {
            waypoints.push(start_point);
        }
        diagnostics.visited_nodes = search.expanded;
        diagnostics.reached_goal = search.reached_goal;
        diagnostics.length = path_length(&waypoints);
        diagnostics.cost = diagnostics.length;

        if search.reached_goal {
            PathSearchOutcome {
                waypoints,
                diagnostics,
                reached_goal: true,
                failure_reason: None,
            }
        } else {
            PathSearchOutcome {
                waypoints,
                diagnostics,
                reached_goal: false,
                failure_reason: Some(PathFailureReason::NoPath),
            }
        }
    }

    fn grid_for_snapshot(&self, snapshot: &EnvironmentSnapshot) -> OccupancyGrid {
        let signature = snapshot_signature(snapshot, self.config.cell_size);
        {
            let cache_guard = self.cached.lock();
            if let Some(entry) = cache_guard.as_ref() {
                if entry.signature == signature && entry.grid.signature == signature {
                    debug!(
                        target: "w1z4rdv1510n::search",
                        width_cells = entry.grid.width_cells,
                        height_cells = entry.grid.height_cells,
                        "reusing cached occupancy grid"
                    );
                    return entry.grid.clone();
                }
            }
        }
        let grid = OccupancyGrid::from_snapshot(snapshot, self.config.cell_size, signature);
        let mut cache_guard = self.cached.lock();
        *cache_guard = Some(CachedGrid {
            signature,
            grid: grid.clone(),
        });
        let blocked_cells = grid.blocked.iter().filter(|cell| **cell > 0).count();
        debug!(
            target: "w1z4rdv1510n::search",
            width_cells = grid.width_cells,
            height_cells = grid.height_cells,
            blocked_cells,
            "rebuilt occupancy grid"
        );
        grid
    }
}

impl Clone for SearchModule {
    fn clone(&self) -> Self {
        SearchModule::new(self.config.clone())
    }
}

fn clamp_with_depth(position: &Position, depth: f64) -> Position {
    Position {
        x: position.x,
        y: position.y,
        z: position.z.clamp(0.0, depth),
    }
}

#[derive(Debug, Clone)]
struct PathSearchOutcome {
    waypoints: Vec<Position>,
    diagnostics: PathDiagnostics,
    reached_goal: bool,
    failure_reason: Option<PathFailureReason>,
}

impl PathSearchOutcome {
    fn blocked(reason: PathFailureReason, start: Position, diagnostics: PathDiagnostics) -> Self {
        Self {
            waypoints: vec![start],
            diagnostics,
            reached_goal: false,
            failure_reason: Some(reason),
        }
    }
}

#[derive(Clone)]
struct OccupancyGrid {
    signature: SnapshotSignature,
    bounds: Bounds,
    cell_size: f64,
    width_cells: usize,
    height_cells: usize,
    blocked: Vec<u8>,
}

impl OccupancyGrid {
    fn from_snapshot(
        snapshot: &EnvironmentSnapshot,
        cell_size: f64,
        signature: SnapshotSignature,
    ) -> Self {
        let bounds = Bounds::from_snapshot(snapshot);
        let cell_size = cell_size.max(0.25);
        let width_cells = ((bounds.width / cell_size).ceil() as usize).max(1);
        let height_cells = ((bounds.height / cell_size).ceil() as usize).max(1);
        let mut grid = Self {
            signature,
            bounds,
            cell_size,
            width_cells,
            height_cells,
            blocked: vec![0u8; width_cells * height_cells],
        };
        for symbol in &snapshot.symbols {
            if is_blocker(symbol) {
                grid.mark_symbol(symbol);
            }
        }
        grid
    }

    fn clamp_position(&self, pos: &Position) -> Position {
        Position {
            x: pos.x.clamp(0.0, self.bounds.width),
            y: pos.y.clamp(0.0, self.bounds.height),
            z: pos.z,
        }
    }

    fn position_to_cell(&self, pos: &Position) -> Option<GridCell> {
        if pos.x.is_nan() || pos.y.is_nan() {
            return None;
        }
        if pos.x < 0.0 || pos.y < 0.0 || pos.x > self.bounds.width || pos.y > self.bounds.height {
            return None;
        }
        let x = (pos.x / self.cell_size)
            .floor()
            .clamp(0.0, (self.width_cells.saturating_sub(1)) as f64) as i32;
        let y = (pos.y / self.cell_size)
            .floor()
            .clamp(0.0, (self.height_cells.saturating_sub(1)) as f64) as i32;
        Some(GridCell { x, y })
    }

    fn index(&self, cell: GridCell) -> Option<usize> {
        if cell.x < 0
            || cell.y < 0
            || cell.x >= self.width_cells as i32
            || cell.y >= self.height_cells as i32
        {
            return None;
        }
        Some(cell.y as usize * self.width_cells + cell.x as usize)
    }

    fn cell_from_index(&self, idx: usize) -> Option<GridCell> {
        if idx >= self.blocked.len() {
            return None;
        }
        let y = idx / self.width_cells;
        let x = idx % self.width_cells;
        Some(GridCell {
            x: x as i32,
            y: y as i32,
        })
    }

    fn is_blocked(&self, cell: GridCell) -> bool {
        self.index(cell)
            .and_then(|idx| self.blocked.get(idx))
            .map(|value| *value != 0)
            .unwrap_or(true)
    }

    fn mark_symbol(&mut self, symbol: &Symbol) {
        let half_x = extent_from_props(symbol, &["width", "length", "size_x"]).unwrap_or(1.0) / 2.0;
        let half_y = extent_from_props(symbol, &["height", "size_y"]).unwrap_or(1.0) / 2.0;
        let min_x = symbol.position.x - half_x;
        let max_x = symbol.position.x + half_x;
        let min_y = symbol.position.y - half_y;
        let max_y = symbol.position.y + half_y;
        let start_x = (min_x / self.cell_size).floor() as i32;
        let end_x = (max_x / self.cell_size).ceil() as i32;
        let start_y = (min_y / self.cell_size).floor() as i32;
        let end_y = (max_y / self.cell_size).ceil() as i32;
        for y in start_y..=end_y {
            for x in start_x..=end_x {
                let cell = GridCell { x, y };
                if let Some(idx) = self.index(cell) {
                    self.blocked[idx] = 1;
                }
            }
        }
    }

    fn nearest_free_cell(&self, start: GridCell) -> Option<GridCell> {
        let mut visited = vec![false; self.blocked.len()];
        let mut queue = VecDeque::new();
        if let Some(idx) = self.index(start) {
            if self.blocked[idx] == 0 {
                return Some(start);
            }
            visited[idx] = true;
            queue.push_back(start);
        } else {
            return None;
        }
        const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        while let Some(cell) = queue.pop_front() {
            for (dx, dy) in DIRS {
                let neighbor = GridCell {
                    x: cell.x + dx,
                    y: cell.y + dy,
                };
                let Some(idx) = self.index(neighbor) else {
                    continue;
                };
                if visited[idx] {
                    continue;
                }
                visited[idx] = true;
                if self.blocked[idx] == 0 {
                    return Some(neighbor);
                }
                queue.push_back(neighbor);
            }
        }
        None
    }

    fn neighbors(&self, cell: GridCell) -> Vec<(GridCell, f64)> {
        const DIRS: [(i32, i32, f64); 8] = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, std::f64::consts::SQRT_2),
            (1, -1, std::f64::consts::SQRT_2),
            (-1, 1, std::f64::consts::SQRT_2),
            (-1, -1, std::f64::consts::SQRT_2),
        ];
        let mut neighbors = Vec::with_capacity(8);
        for (dx, dy, cost) in DIRS {
            let next = GridCell {
                x: cell.x + dx,
                y: cell.y + dy,
            };
            if self.is_blocked(next) {
                continue;
            }
            if dx != 0 && dy != 0 {
                let adj_a = GridCell {
                    x: cell.x + dx,
                    y: cell.y,
                };
                let adj_b = GridCell {
                    x: cell.x,
                    y: cell.y + dy,
                };
                if self.is_blocked(adj_a) || self.is_blocked(adj_b) {
                    continue;
                }
            }
            if self.index(next).is_some() {
                neighbors.push((next, cost));
            }
        }
        neighbors
    }

    fn cell_center(&self, cell: GridCell, z: f64) -> Position {
        Position {
            x: ((cell.x as f64 + 0.5) * self.cell_size).min(self.bounds.width),
            y: ((cell.y as f64 + 0.5) * self.cell_size).min(self.bounds.height),
            z,
        }
    }

    fn cells_to_waypoints(
        &self,
        cells: &[GridCell],
        start: &Position,
        goal: &Position,
        reached_goal: bool,
    ) -> Vec<Position> {
        if cells.is_empty() {
            return vec![*start];
        }
        let mut points = Vec::with_capacity(cells.len() + 1);
        points.push(*start);
        for (idx, cell) in cells.iter().enumerate().skip(1) {
            let is_last = idx == cells.len() - 1;
            if is_last && reached_goal {
                points.push(*goal);
            } else {
                points.push(self.cell_center(*cell, goal.z));
            }
        }
        if reached_goal
            && (points
                .last()
                .map(|p| !approx_equal(p, goal))
                .unwrap_or(true))
        {
            points.push(*goal);
        } else if !reached_goal {
            let last_cell = cells.last().copied().unwrap_or(cells[0]);
            let projected = self.cell_center(last_cell, goal.z);
            if points
                .last()
                .map(|p| !approx_equal(p, &projected))
                .unwrap_or(true)
            {
                points.push(projected);
            }
        }
        points
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GridCell {
    x: i32,
    y: i32,
}

#[derive(Clone, Copy)]
struct Bounds {
    width: f64,
    height: f64,
    depth: f64,
}

impl Bounds {
    fn from_snapshot(snapshot: &EnvironmentSnapshot) -> Self {
        let width = snapshot
            .bounds
            .get("width")
            .copied()
            .unwrap_or_else(|| {
                snapshot
                    .symbols
                    .iter()
                    .map(|s| s.position.x)
                    .fold(1.0, f64::max)
                    + 1.0
            })
            .max(1.0);
        let height = snapshot
            .bounds
            .get("height")
            .copied()
            .unwrap_or_else(|| {
                snapshot
                    .symbols
                    .iter()
                    .map(|s| s.position.y)
                    .fold(1.0, f64::max)
                    + 1.0
            })
            .max(1.0);
        let depth = snapshot
            .bounds
            .get("depth")
            .copied()
            .unwrap_or(f64::INFINITY);
        Self {
            width,
            height,
            depth,
        }
    }
}

struct AStarResult {
    path_cells: Vec<GridCell>,
    reached_goal: bool,
    expanded: usize,
}

fn run_a_star(grid: &OccupancyGrid, start: GridCell, goal: GridCell) -> AStarResult {
    if start == goal {
        return AStarResult {
            path_cells: vec![start],
            reached_goal: true,
            expanded: 0,
        };
    }
    let mut open = BinaryHeap::new();
    open.push(OpenNode {
        cell: start,
        f_score: heuristic(start, goal),
        g_score: 0.0,
    });
    let mut came_from = vec![None; grid.blocked.len()];
    let mut g_score = vec![f64::INFINITY; grid.blocked.len()];
    if let Some(idx) = grid.index(start) {
        g_score[idx] = 0.0;
    }
    let mut expanded = 0usize;
    let mut best_cell = start;
    let mut best_estimate = heuristic(start, goal);

    while let Some(current) = open.pop() {
        if current.cell == goal {
            let path = reconstruct_path(&came_from, current.cell, grid);
            return AStarResult {
                path_cells: path,
                reached_goal: true,
                expanded,
            };
        }
        expanded += 1;

        for (neighbor, step_cost) in grid.neighbors(current.cell) {
            let Some(idx) = grid.index(neighbor) else {
                continue;
            };
            let tentative_g = current.g_score + step_cost;
            if tentative_g + 1e-9 >= g_score[idx] {
                continue;
            }
            came_from[idx] = Some(current.cell);
            g_score[idx] = tentative_g;
            let f = tentative_g + heuristic(neighbor, goal);
            open.push(OpenNode {
                cell: neighbor,
                f_score: f,
                g_score: tentative_g,
            });
            let estimate = heuristic(neighbor, goal);
            if estimate < best_estimate {
                best_estimate = estimate;
                best_cell = neighbor;
            }
        }
        let estimate = heuristic(current.cell, goal);
        if estimate < best_estimate {
            best_estimate = estimate;
            best_cell = current.cell;
        }
    }

    let path = reconstruct_path(&came_from, best_cell, grid);
    AStarResult {
        path_cells: path,
        reached_goal: false,
        expanded,
    }
}

fn reconstruct_path(
    came_from: &[Option<GridCell>],
    mut current: GridCell,
    grid: &OccupancyGrid,
) -> Vec<GridCell> {
    let mut path = vec![current];
    while let Some(idx) = grid.index(current) {
        if let Some(prev) = came_from[idx] {
            current = prev;
            path.push(current);
        } else {
            break;
        }
    }
    path.reverse();
    path
}

#[derive(Debug, Clone, Copy)]
struct OpenNode {
    cell: GridCell,
    f_score: f64,
    g_score: f64,
}

impl PartialEq for OpenNode {
    fn eq(&self, other: &Self) -> bool {
        self.cell.x == other.cell.x && self.cell.y == other.cell.y
    }
}

impl Eq for OpenNode {}

impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other
            .f_score
            .partial_cmp(&self.f_score)
            .or_else(|| other.g_score.partial_cmp(&self.g_score))
    }
}

impl Ord for OpenNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn heuristic(a: GridCell, b: GridCell) -> f64 {
    let dx = (a.x - b.x) as f64;
    let dy = (a.y - b.y) as f64;
    (dx * dx + dy * dy).sqrt()
}

fn extent_from_props(symbol: &Symbol, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| symbol.properties.get(*key))
        .and_then(Value::as_f64)
        .map(|value| value.max(0.1))
}

fn symbol_radius(symbol: &Symbol) -> f64 {
    extent_from_props(symbol, &["radius", "collision_radius"])
        .or_else(|| {
            extent_from_props(symbol, &["width", "length", "size_x"]).map(|extent| extent / 2.0)
        })
        .unwrap_or(0.5)
}

fn hash_bounds(hasher: &mut DefaultHasher, bounds: &HashMap<String, f64>) {
    if bounds.is_empty() {
        return;
    }
    let mut entries: Vec<_> = bounds.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in entries {
        key.hash(hasher);
        hasher.write_u64(value.to_bits());
    }
}

fn hash_metadata(hasher: &mut DefaultHasher, metadata: &HashMap<String, Value>) {
    if metadata.is_empty() {
        return;
    }
    let mut entries: Vec<_> = metadata.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in entries {
        key.hash(hasher);
        hash_json_value(hasher, value);
    }
}

fn hash_json_value(hasher: &mut DefaultHasher, value: &Value) {
    match value {
        Value::Null => {
            0u8.hash(hasher);
        }
        Value::Bool(flag) => {
            1u8.hash(hasher);
            flag.hash(hasher);
        }
        Value::Number(num) => {
            2u8.hash(hasher);
            if let Some(float) = num.as_f64() {
                hasher.write_u64(float.to_bits());
            } else if let Some(int_val) = num.as_i64() {
                int_val.hash(hasher);
            } else if let Some(u_val) = num.as_u64() {
                u_val.hash(hasher);
            }
        }
        Value::String(text) => {
            3u8.hash(hasher);
            text.hash(hasher);
        }
        Value::Array(items) => {
            4u8.hash(hasher);
            for item in items {
                hash_json_value(hasher, item);
            }
        }
        Value::Object(map) => {
            5u8.hash(hasher);
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (key, value) in entries {
                key.hash(hasher);
                hash_json_value(hasher, value);
            }
        }
    }
}

fn hash_symbol_type(hasher: &mut DefaultHasher, symbol_type: &SymbolType) {
    let tag = match symbol_type {
        SymbolType::Person => 1u8,
        SymbolType::Object => 2u8,
        SymbolType::Wall => 3u8,
        SymbolType::Exit => 4u8,
        SymbolType::Custom => 5u8,
    };
    hasher.write_u8(tag);
}

fn hash_position(hasher: &mut DefaultHasher, position: &Position) {
    hasher.write_u64(position.x.to_bits());
    hasher.write_u64(position.y.to_bits());
    hasher.write_u64(position.z.to_bits());
}

fn hash_blocker_extents(hasher: &mut DefaultHasher, symbol: &Symbol) {
    let width = extent_from_props(symbol, &["width", "length", "size_x"]).unwrap_or(1.0);
    let height = extent_from_props(symbol, &["height", "size_y"]).unwrap_or(1.0);
    let depth = extent_from_props(symbol, &["depth", "size_z"]).unwrap_or(0.0);
    let radius = symbol_radius(symbol);
    hasher.write_u64(width.to_bits());
    hasher.write_u64(height.to_bits());
    hasher.write_u64(depth.to_bits());
    hasher.write_u64(radius.to_bits());
}

fn hash_blocker_flags(hasher: &mut DefaultHasher, props: &HashMap<String, Value>) {
    for key in ["walkable", "blocking", "solid", "layer", "material"] {
        if let Some(value) = props.get(key) {
            key.hash(hasher);
            hash_json_value(hasher, value);
        }
    }
}

#[derive(Clone)]
struct CachedGrid {
    signature: SnapshotSignature,
    grid: OccupancyGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotSignature {
    hash: u64,
    blocker_digest: u64,
}

fn snapshot_signature(snapshot: &EnvironmentSnapshot, cell_size: f64) -> SnapshotSignature {
    let mut base_hasher = DefaultHasher::new();
    snapshot.timestamp.unix.hash(&mut base_hasher);
    (snapshot.symbols.len() as u64).hash(&mut base_hasher);
    cell_size.to_bits().hash(&mut base_hasher);
    hash_bounds(&mut base_hasher, &snapshot.bounds);
    hash_metadata(&mut base_hasher, &snapshot.metadata);

    let mut blocker_hasher = DefaultHasher::new();
    let mut blockers: Vec<_> = snapshot
        .symbols
        .iter()
        .filter(|sym| is_blocker(sym))
        .collect();
    blockers.sort_by(|a, b| a.id.cmp(&b.id));
    for symbol in blockers {
        symbol.id.hash(&mut blocker_hasher);
        hash_symbol_type(&mut blocker_hasher, &symbol.symbol_type);
        hash_position(&mut blocker_hasher, &symbol.position);
        hash_blocker_extents(&mut blocker_hasher, symbol);
        hash_blocker_flags(&mut blocker_hasher, &symbol.properties);
    }

    SnapshotSignature {
        hash: base_hasher.finish(),
        blocker_digest: blocker_hasher.finish(),
    }
}

fn is_blocker(symbol: &Symbol) -> bool {
    if matches!(symbol.symbol_type, SymbolType::Wall) {
        return true;
    }
    if let Some(Value::Bool(false)) = symbol.properties.get("walkable") {
        return true;
    }
    if let Some(Value::Bool(true)) = symbol.properties.get("blocking") {
        return true;
    }
    if let Some(Value::Bool(true)) = symbol.properties.get("solid") {
        return true;
    }
    false
}

fn approx_equal(a: &Position, b: &Position) -> bool {
    (a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6 && (a.z - b.z).abs() < 1e-6
}

fn path_length(points: &[Position]) -> f64 {
    points
        .windows(2)
        .map(|pair| {
            let dx = pair[0].x - pair[1].x;
            let dy = pair[0].y - pair[1].y;
            let dz = pair[0].z - pair[1].z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Properties, SymbolState, SymbolType};

    fn bounds(width: f64, height: f64) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("width".into(), width);
        map.insert("height".into(), height);
        map
    }

    fn person(id: &str, position: Position) -> Symbol {
        Symbol {
            id: id.into(),
            symbol_type: SymbolType::Person,
            position,
            properties: Properties::new(),
        }
    }

    fn wall(id: &str, position: Position, width: f64, height: f64) -> Symbol {
        let mut props = Properties::new();
        props.insert("width".into(), Value::from(width));
        props.insert("height".into(), Value::from(height));
        Symbol {
            id: id.into(),
            symbol_type: SymbolType::Wall,
            position,
            properties: props,
        }
    }

    #[test]
    fn enforce_constraints_moves_agent_out_of_blocked_cell() {
        let snapshot = EnvironmentSnapshot {
            timestamp: crate::schema::Timestamp { unix: 0 },
            bounds: bounds(4.0, 4.0),
            symbols: vec![
                person(
                    "agent",
                    Position {
                        x: 0.5,
                        y: 0.5,
                        z: 0.0,
                    },
                ),
                wall(
                    "wall",
                    Position {
                        x: 2.0,
                        y: 2.0,
                        z: 0.0,
                    },
                    2.0,
                    2.0,
                ),
            ],
            metadata: Properties::new(),
            stack_history: Vec::new(),
        };
        let mut state = DynamicState {
            timestamp: snapshot.timestamp,
            symbol_states: HashMap::from([(
                "agent".into(),
                SymbolState {
                    position: Position {
                        x: 2.0,
                        y: 2.0,
                        z: 0.0,
                    },
                    ..Default::default()
                },
            )]),
        };
        let module = SearchModule::new(SearchConfig {
            cell_size: 0.5,
            teleport_on_no_path: true,
        });
        module.enforce_hard_constraints(&snapshot, &mut state, None);
        let repaired = state.symbol_states.get("agent").unwrap().position;
        let grid = module.grid_for_snapshot(&snapshot);
        let cell = grid.position_to_cell(&repaired).expect("valid cell");
        assert!(
            !grid.is_blocked(cell),
            "agent should no longer be inside wall"
        );
    }

    #[test]
    fn enforce_constraints_separates_overlapping_agents() {
        let snapshot = EnvironmentSnapshot {
            timestamp: crate::schema::Timestamp { unix: 0 },
            bounds: bounds(3.0, 3.0),
            symbols: vec![
                person(
                    "a",
                    Position {
                        x: 0.5,
                        y: 0.5,
                        z: 0.0,
                    },
                ),
                person(
                    "b",
                    Position {
                        x: 1.5,
                        y: 0.5,
                        z: 0.0,
                    },
                ),
            ],
            metadata: Properties::new(),
            stack_history: Vec::new(),
        };
        let overlapping = Position {
            x: 1.0,
            y: 1.0,
            z: 0.0,
        };
        let mut state = DynamicState {
            timestamp: snapshot.timestamp,
            symbol_states: HashMap::from([
                (
                    "a".into(),
                    SymbolState {
                        position: overlapping,
                        ..Default::default()
                    },
                ),
                (
                    "b".into(),
                    SymbolState {
                        position: overlapping,
                        ..Default::default()
                    },
                ),
            ]),
        };
        let module = SearchModule::new(SearchConfig::default());
        module.enforce_hard_constraints(&snapshot, &mut state, None);
        let a_pos = state.symbol_states.get("a").unwrap().position;
        let b_pos = state.symbol_states.get("b").unwrap().position;
        let dx = a_pos.x - b_pos.x;
        let dy = a_pos.y - b_pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(
            dist >= 0.9,
            "agents should be separated beyond combined radii, got {dist}"
        );
    }
}
