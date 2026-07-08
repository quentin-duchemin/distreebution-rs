use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// FenwickTree
// ─────────────────────────────────────────────────────────────────────────────
#[pyclass]
pub struct FenwickTree {
    n: usize,
    v: Vec<f64>,
}

#[pymethods]
impl FenwickTree {
    #[new]
    pub fn new(n: usize) -> Self { FenwickTree { n, v: vec![0.0; n] } }
    pub fn __len__(&self) -> usize { self.n }

    pub fn prefix_sum(&self, stop: usize) -> PyResult<f64> {
        if stop > self.n { return Err(PyValueError::new_err("index out of range")); }
        let mut s = 0.0f64;
        let mut i = stop as isize;
        while i > 0 { s += self.v[(i-1) as usize]; i &= i-1; }
        Ok(s)
    }

    pub fn range_sum(&self, start: usize, stop: usize) -> PyResult<f64> {
        if stop < start || stop > self.n { return Err(PyValueError::new_err("index out of range")); }
        if stop == start { return Ok(0.0); }
        let mut r = self.prefix_sum(stop)?;
        if start > 0 { r -= self.prefix_sum(start)?; }
        Ok(r)
    }

    pub fn add(&mut self, idx: usize, k: f64) -> PyResult<()> {
        if idx >= self.n { return Err(PyValueError::new_err("index out of range")); }
        let mut i = (idx + 1) as isize;
        while i <= self.n as isize { self.v[(i-1) as usize] += k; i += i & -i; }
        Ok(())
    }

    pub fn __getitem__(&self, idx: usize) -> PyResult<f64> { self.range_sum(idx, idx+1) }

    pub fn init(&mut self, frequencies: Vec<f64>) -> PyResult<()> {
        if frequencies.len() != self.n { return Err(PyValueError::new_err("length mismatch")); }
        self.v = frequencies;
        for idx in 1..=self.n {
            let parent = idx + (idx & idx.wrapping_neg());
            if parent <= self.n { let val = self.v[idx-1]; self.v[parent-1] += val; }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MinMaxHeap internals (standalone functions)
// ─────────────────────────────────────────────────────────────────────────────
fn heap_level(i: usize) -> u32 { (i+1).ilog2() }

fn trickle_down_min(a: &mut Vec<f64>, i: usize, size: usize) {
    if size <= i*2+1 { return; }
    let mut m = i*2+1;
    if i*2+2 < size && a[i*2+2] < a[m] { m = i*2+2; }
    let mut child = true;
    for j in (i*4+3)..(i*4+7).min(size) { if a[j] < a[m] { m = j; child = false; } }
    if child { if a[m] < a[i] { a.swap(i, m); } }
    else if a[m] < a[i] {
        a.swap(m, i);
        let p = (m-1)/2;
        if a[m] > a[p] { a.swap(m, p); }
        trickle_down_min(a, m, size);
    }
}

fn trickle_down_max(a: &mut Vec<f64>, i: usize, size: usize) {
    if size <= i*2+1 { return; }
    let mut m = i*2+1;
    if i*2+2 < size && a[i*2+2] > a[m] { m = i*2+2; }
    let mut child = true;
    for j in (i*4+3)..(i*4+7).min(size) { if a[j] > a[m] { m = j; child = false; } }
    if child { if a[m] > a[i] { a.swap(i, m); } }
    else if a[m] > a[i] {
        a.swap(m, i);
        let p = (m-1)/2;
        if a[m] < a[p] { a.swap(m, p); }
        trickle_down_max(a, m, size);
    }
}

fn trickle_down(a: &mut Vec<f64>, i: usize, size: usize) {
    if heap_level(i)%2==0 { trickle_down_min(a,i,size); } else { trickle_down_max(a,i,size); }
}

fn bubble_up_min(a: &mut Vec<f64>, mut i: usize) {
    while i > 2 { let gp=(i.wrapping_sub(3))/4; if a[i]<a[gp] { a.swap(i,gp); i=gp; } else { return; } }
}
fn bubble_up_max(a: &mut Vec<f64>, mut i: usize) {
    while i > 2 { let gp=(i.wrapping_sub(3))/4; if a[i]>a[gp] { a.swap(i,gp); i=gp; } else { return; } }
}

fn heap_insert(a: &mut Vec<f64>, k: f64, size: usize) {
    if size >= a.len() { a.push(k); } else { a[size] = k; }
    let i = size;
    if heap_level(i)%2==0 {
        if i > 0 && a[i] > a[(i-1)/2] { a.swap(i,(i-1)/2); bubble_up_max(a,(i-1)/2); }
        else { bubble_up_min(a,i); }
    } else {
        if i > 0 && a[i] < a[(i-1)/2] { a.swap(i,(i-1)/2); bubble_up_min(a,(i-1)/2); }
        else { bubble_up_max(a,i); }
    }
}

fn peek_min(a: &[f64], size: usize) -> Option<f64> { if size==0 { None } else { Some(a[0]) } }
fn peek_max(a: &[f64], size: usize) -> Option<f64> {
    match size { 0=>None, 1=>Some(a[0]), 2=>Some(a[1]), _=>Some(a[1].max(a[2])) }
}

fn remove_min(a: &mut Vec<f64>, size: usize) -> (Option<f64>, usize) {
    if size==0 { return (None,0); }
    let e=a[0]; a[0]=a[size-1]; trickle_down(a,0,size-1); (Some(e),size-1)
}
fn remove_max(a: &mut Vec<f64>, size: usize) -> (Option<f64>, usize) {
    if size==0 { return (None,0); }
    if size==1 { return (Some(a[0]),0); }
    if size==2 { return (Some(a[1]),1); }
    let i = if a[1]>a[2] { 1 } else { 2 };
    let e=a[i]; a[i]=a[size-1]; trickle_down(a,i,size-1); (Some(e),size-1)
}

#[pyclass]
pub struct MinMaxHeap {
    pub a: Vec<f64>,
    pub size: usize,
}

#[pymethods]
impl MinMaxHeap {
    #[new]
    #[pyo3(signature = (reserve=0))]
    pub fn new(reserve: usize) -> Self { MinMaxHeap { a: Vec::with_capacity(reserve), size: 0 } }
    pub fn __len__(&self) -> usize { self.size }
    pub fn insert(&mut self, key: f64) { heap_insert(&mut self.a, key, self.size); self.size += 1; }
    pub fn peekmin(&self) -> Option<f64> { peek_min(&self.a, self.size) }
    pub fn peekmax(&self) -> Option<f64> { peek_max(&self.a, self.size) }
    pub fn popmin(&mut self) -> Option<f64> {
        let (m,s)=remove_min(&mut self.a,self.size); self.size=s;
        if m.is_some() { self.a.truncate(s); } m
    }
    pub fn popmax(&mut self) -> Option<f64> {
        let (m,s)=remove_max(&mut self.a,self.size); self.size=s; self.a.truncate(s); m
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Argsort helper
// ─────────────────────────────────────────────────────────────────────────────


// ─────────────────────────────────────────────────────────────────────────────
// entropies_Quadratic
// ─────────────────────────────────────────────────────────────────────────────
#[pyfunction]
#[pyo3(signature = (order, y, loo=false))]
pub fn entropies_quadratic(order: Vec<usize>, y: Vec<f64>, loo: bool) -> Vec<f64> {
    let n = order.len();
    let mut entropy = vec![0.0f64; n+1];
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for idx in 0..n {
        let yv = y[order[idx]];
        sum += yv; sum_sq += yv*yv;
        let k = (idx+1) as f64;
        let mean = sum/k;
        let var = sum_sq/k - mean*mean;
        entropy[idx+1] = if loo && idx >= 2 {
            let sigma2 = (sum_sq - k*mean*mean)/(k-1.0);
            var + 2.0*sigma2/k
        } else { var };
    }
    entropy[1] = 0.0;
    entropy
}

// ─────────────────────────────────────────────────────────────────────────────
// entropies_CRPS
// ─────────────────────────────────────────────────────────────────────────────
#[pyfunction]
#[pyo3(signature = (order, y, loo=false))]
pub fn entropies_crps(order: Vec<usize>, y: Vec<f64>, loo: bool) -> Vec<f64> {
    let n = order.len();
    let ysort: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let mut out = vec![0.0f64; n + 1];
    let mut bit = Bit::new(n.max(1));
    entropies_crps_fast(&ysort, loo, &mut out, &mut bit);
    out
}



// ═════════════════════════════════════════════════════════════════════════════
// FAST INTERNAL PATH  (no PyResult, no BTreeMap, O(n log n) ranks)
// ═════════════════════════════════════════════════════════════════════════════

/// Plain Fenwick tree over f64, no bounds-checking Result wrapping.
struct Bit {
    v: Vec<f64>,
}
impl Bit {
    #[inline]
    fn new(n: usize) -> Self { Bit { v: vec![0.0; n] } }
    #[inline]
    fn clear(&mut self) { for x in self.v.iter_mut() { *x = 0.0; } }
    #[inline]
    fn prefix_sum(&self, stop: usize) -> f64 {
        let mut s = 0.0;
        let mut i = stop as isize;
        while i > 0 { s += unsafe { *self.v.get_unchecked((i - 1) as usize) }; i &= i - 1; }
        s
    }
    #[inline]
    fn add(&mut self, idx: usize, k: f64) {
        let n = self.v.len();
        let mut i = (idx + 1) as isize;
        while i <= n as isize { unsafe { *self.v.get_unchecked_mut((i - 1) as usize) += k; } i += i & -i; }
    }
}

/// Compute the `pos` array (rank in sorted order, 0-based) and 1-based ranks with
/// tie handling for a slice, in O(n log n), reusing a single sort.
///
/// Returns (pos, ranks_1based) where:
///   pos[i]   = index of ysort[i] in the value-sorted order (stable, ties broken by original index)
///   ranks[i] = number of elements <= ysort[i] among ysort[0..=i]  (matches Python WBTree)
///
/// The ranks are the count of prior-or-equal elements; computed with a value-counting
/// Fenwick over the compressed value domain → O(n log n).
#[inline]
fn ranks_and_pos(ysort: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = ysort.len();
    // argsort by value (stable via index tiebreak)
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let va = ysort[a as usize];
        let vb = ysort[b as usize];
        va.total_cmp(&vb).then(a.cmp(&b))
    });
    // pos[a] = sorted position of element a
    let mut pos = vec![0usize; n];
    for (rank_i, &a) in order.iter().enumerate() {
        pos[a as usize] = rank_i;
    }
    // compressed rank of each element's VALUE (dense 0..distinct) for tie-aware counting
    // value_rank[i] = number of distinct values strictly less than ysort[i]
    let mut value_rank = vec![0u32; n];
    let mut distinct = 0u32;
    for (k, &a) in order.iter().enumerate() {
        if k > 0 {
            let prev = order[k - 1] as usize;
            if ysort[a as usize] != ysort[prev] { distinct += 1; }
        }
        value_rank[a as usize] = distinct;
    }
    let ndistinct = (distinct + 1) as usize;

    // Fenwick over value domain: count elements <= current value seen so far
    let mut cnt = vec![0i64; ndistinct + 1];
    // ranks[i] = (#elements already inserted with value <= ysort[i]) + 1
    let mut ranks = vec![0usize; n];
    for i in 0..n {
        let vr = value_rank[i] as usize; // 0-based value bucket
        // prefix count over buckets [0..=vr]
        let mut c = 0i64;
        let mut j = (vr + 1) as isize;
        while j > 0 { c += cnt[j as usize]; j &= j - 1; }
        ranks[i] = c as usize + 1;
        // add 1 at bucket vr
        let mut j = (vr + 1) as isize;
        while j <= ndistinct as isize { cnt[j as usize] += 1; j += j & -j; }
    }
    (pos, ranks)
}

/// Fast CRPS entropies operating directly on a pre-reordered ysort slice.
/// Writes n+1 values into `out`. `bit` is a scratch Fenwick reused across calls.
#[inline]
fn entropies_crps_fast(ysort: &[f64], loo: bool, out: &mut [f64], bit: &mut Bit) {
    let n = ysort.len();
    out[0] = 0.0;
    if n == 0 { return; }
    out[1] = 0.0;

    let (pos, ranks) = ranks_and_pos(ysort);

    bit.clear();
    bit.add(pos[0], ysort[0]);
    let mut wup = ysort[0];
    let mut s = ysort[0];
    let mut hup = 0.0f64;
    let mut hlow = 0.0f64;

    for idx in 1..n {
        let ni = idx as f64;
        let s_old = s;
        let yi = ysort[idx];
        s += yi;
        let cum0 = bit.prefix_sum(pos[idx]);
        let cum_end = s_old - cum0;
        let r = ranks[idx] as f64;
        wup += r * yi + cum_end;
        hup += -2.0 * s + 2.0 * cum0 + 2.0 * wup + (r - 1.0) * (r - 2.0) * yi;
        hlow += 2.0 * (ni + 1.0) * cum0 + (ni - r + 1.0) * (ni - r + 2.0) * yi;
        let e = (hup - hlow) / (ni + 1.0).powi(3);
        out[idx + 1] = if loo && idx >= 2 { e * (ni + 1.0).powi(2) / ni.powi(2) } else { e };
        bit.add(pos[idx], yi);
    }
}

/// Fast quadratic entropies on a pre-reordered ysort slice.
#[inline]
fn entropies_quadratic_fast(ysort: &[f64], loo: bool, out: &mut [f64]) {
    let n = ysort.len();
    out[0] = 0.0;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for idx in 0..n {
        let yv = ysort[idx];
        sum += yv; sum_sq += yv * yv;
        let k = (idx + 1) as f64;
        let mean = sum / k;
        let var = sum_sq / k - mean * mean;
        out[idx + 1] = if loo && idx >= 2 {
            let sigma2 = (sum_sq - k * mean * mean) / (k - 1.0);
            var + 2.0 * sigma2 / k
        } else { var };
    }
    if n >= 1 { out[1] = 0.0; }
}


// entropies_MultiQuantiles
// ─────────────────────────────────────────────────────────────────────────────
fn level2idx(n: usize, q: f64) -> usize {
    let v = (n as f64 * q).ceil() as isize - 1;
    v.max(0).min(n as isize - 1) as usize
}

fn get_entropy_vals(values: &[f64], quantiles: &[f64]) -> f64 {
    let mut vals = values.to_vec();
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len(); let tot: f64 = vals.iter().sum();
    let mut cum = 0.0f64; let mut prev = 0usize; let mut res = 0.0f64;
    for &q in quantiles {
        let fqn = level2idx(n,q);
        for i in prev..=fqn { cum += vals[i]; } prev = fqn+1;
        res += vals[fqn]*((fqn+1) as f64/n as f64 - q) + tot*q/n as f64 - cum/n as f64;
    }
    res
}

struct HeapGroup { a: Vec<f64>, size: usize, sum: f64 }
impl HeapGroup {
    fn new() -> Self { HeapGroup { a: Vec::new(), size: 0, sum: 0.0 } }
    fn insert(&mut self, v: f64) { heap_insert(&mut self.a, v, self.size); self.size+=1; self.sum+=v; }
    fn popmax(&mut self) -> Option<f64> {
        let (m,s)=remove_max(&mut self.a,self.size); self.size=s; self.a.truncate(s);
        if let Some(v)=m { self.sum-=v; } m
    }
    fn popmin(&mut self) -> Option<f64> {
        let (m,s)=remove_min(&mut self.a,self.size); self.size=s; self.a.truncate(s);
        if let Some(v)=m { self.sum-=v; } m
    }
    fn peekmax(&self) -> Option<f64> { peek_max(&self.a,self.size) }
    fn peekmin(&self) -> Option<f64> { peek_min(&self.a,self.size) }
}

fn hg_get_max_minus(i: isize, heaps: &[HeapGroup]) -> (isize, Option<f64>) {
    let mut j=i;
    while j>=0 { let v=heaps[j as usize].peekmax(); if v.is_some() { return (j,v); } j-=1; }
    (j, None)
}
fn hg_get_min_plus(i: usize, heaps: &[HeapGroup]) -> (usize, Option<f64>) {
    let mut j=i;
    while j<heaps.len() { let v=heaps[j].peekmin(); if v.is_some() { return (j,v); } j+=1; }
    (j, None)
}

fn get_entropy_mq(heaps: &[HeapGroup], quantiles: &[f64], n: usize) -> f64 {
    let tot: f64 = heaps.iter().map(|h| h.sum).sum();
    let mut cum = 0.0f64; let mut res = 0.0f64;
    for (i_q,&q) in quantiles.iter().enumerate() {
        cum += heaps[i_q].sum;
        let (_,mv)=hg_get_max_minus(i_q as isize, heaps);
        if let Some(v)=mv { res += v*((level2idx(n,q)+1) as f64/n as f64 - q); }
        res += tot*q/n as f64 - cum/n as f64;
    }
    res
}

fn loo_mq(ysort: &[f64], entropy: f64, quantiles: &[f64], heaps: &[HeapGroup]) -> f64 {
    let n = ysort.len();
    let mut hloo = entropy * n as f64;
    for (i_q,&q) in quantiles.iter().enumerate() {
        let rstar = level2idx(n,q);
        let (i_qs, y_rs_opt)=hg_get_max_minus(i_q as isize, heaps);
        let y_rs = match y_rs_opt { Some(v)=>v, None=>continue };
        let (_,y_rsp1_opt)=hg_get_min_plus(i_q+1, heaps);
        let y_rsp1 = y_rsp1_opt.unwrap_or(y_rs);
        let y_rsm1 = if heaps[i_qs as usize].size==1 {
            if i_q!=0 { let (_,v)=hg_get_max_minus(i_q as isize-1,heaps); v.unwrap_or(y_rs) }
            else { y_rs }
        } else if heaps[i_qs as usize].size==2 {
            heaps[i_qs as usize].peekmin().unwrap_or(y_rs)
        } else {
            let h=&heaps[i_qs as usize];
            if h.size>2 { h.a[1].min(h.a[2]) } else { y_rs }
        };
        if rstar==level2idx(n-1,q) { hloo += (1.0-q)*(rstar+1) as f64*(y_rsp1-y_rs); }
        else { hloo += q*(n-rstar) as f64*(y_rs-y_rsm1); }
    }
    hloo/n as f64
}

#[pyfunction]
#[pyo3(signature = (order, y, quantiles, loo=false))]
pub fn entropies_multi_quantiles(order: Vec<usize>, y: Vec<f64>, quantiles: Vec<f64>, loo: bool) -> Vec<f64> {
    let n=order.len(); let nq=quantiles.len();
    let mut heaps: Vec<HeapGroup>=(0..=nq).map(|_| HeapGroup::new()).collect();
    let mut heap2size=vec![0usize; nq+1];
    let ysort: Vec<f64>=order.iter().map(|&i| y[i]).collect();
    let mut entropy=vec![0.0f64; n+1];

    heaps[0].insert(ysort[0]); heap2size[0]=1;
    entropy[1]=get_entropy_vals(&[ysort[0]], &quantiles);

    for idx in 1..n {
        let mut y_c=ysort[idx];
        let mut placed=false;
        'outer: for i_q in 0..nq {
            let q=quantiles[i_q];
            let left_size: usize=heap2size[..=i_q].iter().sum();
            let (_,mhq)=hg_get_max_minus(i_q as isize, &heaps);
            if left_size==level2idx(idx+1,q)+1 {
                if let Some(mv)=mhq { if y_c<=mv {
                    let popped=heaps[i_q].popmax().unwrap();
                    heaps[i_q].insert(y_c); y_c=popped;
                }}
            } else {
                let (i_qp,min_plus)=hg_get_min_plus(i_q+1, &heaps);
                if min_plus.is_none() {
                    heaps[i_q].insert(y_c); heap2size[i_q]+=1; placed=true; break 'outer;
                } else {
                    let mpv=min_plus.unwrap();
                    if y_c<=mpv {
                        heaps[i_q].insert(y_c); heap2size[i_q]+=1;
                        let p=heaps[i_qp].popmin().unwrap();
                        heap2size[i_qp]-=1; y_c=p;
                    } else {
                        let p=heaps[i_qp].popmin().unwrap();
                        heap2size[i_qp]-=1;
                        heaps[i_q].insert(p); heap2size[i_q]+=1;
                    }
                }
            }
        }
        if !placed { heaps[nq].insert(y_c); heap2size[nq]+=1; }

        let e=get_entropy_mq(&heaps, &quantiles, idx+1);
        entropy[idx+1] = if loo && idx>=2 { loo_mq(&ysort[..=idx], e, &quantiles, &heaps) } else { e };
    }
    entropy
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree (arena-based)
// ─────────────────────────────────────────────────────────────────────────────
enum NodeData {
    Leaf(Vec<f64>),
    Internal { feat: usize, thr: f64, left: usize, right: usize },
}

struct Arena { nodes: Vec<NodeData> }
impl Arena {
    fn new() -> Self { Arena { nodes: Vec::new() } }
    fn alloc(&mut self, d: NodeData) -> usize { let id=self.nodes.len(); self.nodes.push(d); id }
}

/// Which entropy criterion to use (avoids boxed closures + Vec allocations per call).
#[derive(Clone, Copy)]
enum Criterion {
    Quadratic { loo: bool },
    Crps { loo: bool },
}

/// Column-major feature matrix: cols[f] is the contiguous column for feature f.
struct ColMajor {
    n_rows: usize,
    n_cols: usize,
    cols: Vec<Vec<f64>>,
}
impl ColMajor {
    fn from_rows(rows: &[Vec<f64>]) -> Self {
        let n_rows = rows.len();
        let n_cols = if n_rows == 0 { 0 } else { rows[0].len() };
        let mut cols = vec![Vec::with_capacity(n_rows); n_cols];
        for row in rows {
            for f in 0..n_cols {
                cols[f].push(row[f]);
            }
        }
        ColMajor { n_rows, n_cols, cols }
    }
}

/// Scratch buffers reused across the whole fit to avoid per-node allocation.
/// One set per rayon worker thread.
struct Scratch {
    order: Vec<u32>,
    ysort: Vec<f64>,
    eu: Vec<f64>,
    ed: Vec<f64>,
    bit: Bit,
}
impl Scratch {
    fn new(n: usize) -> Self {
        Scratch {
            order: Vec::with_capacity(n),
            ysort: vec![0.0; n],
            eu: vec![0.0; n + 1],
            ed: vec![0.0; n + 1],
            bit: Bit::new(n.max(1)),
        }
    }
}

/// Evaluate the best split for ONE feature over the current node's sample subset.
/// `sample_idx` are indices into the full column-major matrix that belong to this node.
/// Returns (score, threshold) of the best valid split for this feature, or None.
fn best_split_for_feature(
    col: &[f64],
    sample_idx: &[u32],
    y: &[f64],
    crit: Criterion,
    min_ss: usize,
    require_decrease: bool,
    sc: &mut Scratch,
) -> Option<(f64, f64)> {
    let ns = sample_idx.len();
    if ns <= 1 { return None; }

    // Sort local samples by this feature's value.
    sc.order.clear();
    sc.order.extend_from_slice(sample_idx);
    sc.order.sort_unstable_by(|&a, &b| {
        col[a as usize].total_cmp(&col[b as usize])
    });

    // Build ysort in feature-sorted order.
    for (k, &a) in sc.order.iter().enumerate() {
        sc.ysort[k] = y[a as usize];
    }
    let ys = &sc.ysort[..ns];

    // Entropies up (forward) and down (reverse).
    match crit {
        Criterion::Quadratic { loo } => {
            entropies_quadratic_fast(ys, loo, &mut sc.eu[..ns + 1]);
            // reverse into a temp region of ed's tail then compute
            // Reuse ysort reversed:
            // build reversed ysort in-place using ed as scratch is messy; use small copy
            let mut rev = [0.0f64; 0]; let _ = &mut rev;
            // reverse ysort into a stack of ed? We'll just reverse a local copy.
            // To avoid alloc, reverse ys into sc.ed[..ns] region temporarily is unsafe
            // (aliasing with output). Use a dedicated reverse buffer.
            reverse_and_quad(ys, loo, &mut sc.ed[..ns + 1]);
        }
        Criterion::Crps { loo } => {
            entropies_crps_fast(ys, loo, &mut sc.eu[..ns + 1], &mut sc.bit);
            reverse_and_crps(ys, loo, &mut sc.ed[..ns + 1], &mut sc.bit);
        }
    }

    let len = ns + 1;
    let eu = &sc.eu[..len];
    let ed = &sc.ed[..len];

    // score[i] = i*eu[i] + (len-1-i)*ed[len-1-i]   (weights are 0..len-1)
    // global score = score[len-1]
    let global_score = (len as f64 - 1.0) * eu[len - 1]; // ed[0]=0 contributes 0
    // Find the minimum score among interior split points i=1..len-1, tracking the best
    // valid one (distinct feature values + min_samples_split on both sides).
    // In sorted order, #left = i exactly, so the min_ss check is O(1).
    let mut best: Option<(f64, f64)> = None;
    for i in 1..len - 1 {
        // left count = i, right count = ns - i
        if i < min_ss || ns - i < min_ss { continue; }
        let xa = col[sc.order[i - 1] as usize];
        let xb = col[sc.order[i] as usize];
        if (xa - xb).abs() <= 1e-5 { continue; } // need distinct values to split
        let score = (i as f64) * eu[i] + ((len - 1 - i) as f64) * ed[len - 1 - i];
        if require_decrease && !(score < global_score) { continue; }
        match best {
            None => best = Some((score, (xa + xb) / 2.0)),
            Some((bs, _)) if score < bs => best = Some((score, (xa + xb) / 2.0)),
            _ => {}
        }
    }
    best
}

fn reverse_and_quad(ys: &[f64], loo: bool, out: &mut [f64]) {
    let n = ys.len();
    // compute quadratic entropies on the reversed sequence without allocating:
    // we can run the same accumulator but iterate ys in reverse.
    out[0] = 0.0;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for idx in 0..n {
        let yv = ys[n - 1 - idx];
        sum += yv; sum_sq += yv * yv;
        let k = (idx + 1) as f64;
        let mean = sum / k;
        let var = sum_sq / k - mean * mean;
        out[idx + 1] = if loo && idx >= 2 {
            let sigma2 = (sum_sq - k * mean * mean) / (k - 1.0);
            var + 2.0 * sigma2 / k
        } else { var };
    }
    if n >= 1 { out[1] = 0.0; }
}

fn reverse_and_crps(ys: &[f64], loo: bool, out: &mut [f64], bit: &mut Bit) {
    let n = ys.len();
    // Build reversed ysort on the stack-free path: we need a contiguous reversed slice.
    // Use a thread-local-ish small vec. Allocation here is O(n) once per feature; the
    // dominant cost is the O(n log n) rank build inside entropies_crps_fast, so this
    // extra O(n) copy is negligible.
    let mut rev = vec![0.0f64; n];
    for i in 0..n { rev[i] = ys[n - 1 - i]; }
    entropies_crps_fast(&rev, loo, out, bit);
}

/// Recursively fit the tree. Parallelises the feature loop with rayon at nodes
/// that are large enough to be worth the overhead.
fn fit_tree_cm(
    arena: &mut Arena,
    cm: &ColMajor,
    y: &[f64],
    sample_idx: Vec<u32>,
    depth: usize,
    max_depth: Option<usize>,
    min_ss: usize,
    crit: Criterion,
    require_decrease: bool,
) -> usize {
    let ns = sample_idx.len();
    let stop = max_depth.map(|md| depth >= md).unwrap_or(false) || ns < min_ss || ns <= 1;
    if stop {
        let yv: Vec<f64> = sample_idx.iter().map(|&i| y[i as usize]).collect();
        return arena.alloc(NodeData::Leaf(yv));
    }

    // Evaluate every feature; parallel when the node is large.
    const PAR_THRESHOLD: usize = 512;
    let nf = cm.n_cols;

    let eval_feat = |feat: usize, sc: &mut Scratch| -> Option<(f64, f64, usize)> {
        best_split_for_feature(
            &cm.cols[feat], &sample_idx, y, crit, min_ss, require_decrease, sc,
        ).map(|(score, thr)| (score, thr, feat))
    };

    let best: Option<(f64, f64, usize)> = if ns >= PAR_THRESHOLD {
        (0..nf).into_par_iter()
            .map_init(|| Scratch::new(ns), |sc, feat| eval_feat(feat, sc))
            .flatten()
            .reduce_with(|a, b| if a.0 <= b.0 { a } else { b })
    } else {
        let mut sc = Scratch::new(ns);
        let mut acc: Option<(f64, f64, usize)> = None;
        for feat in 0..nf {
            if let Some(cand) = eval_feat(feat, &mut sc) {
                match acc {
                    None => acc = Some(cand),
                    Some(a) if cand.0 < a.0 => acc = Some(cand),
                    _ => {}
                }
            }
        }
        acc
    };

    match best {
        None => {
            let yv: Vec<f64> = sample_idx.iter().map(|&i| y[i as usize]).collect();
            arena.alloc(NodeData::Leaf(yv))
        }
        Some((_score, thr, feat)) => {
            let col = &cm.cols[feat];
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &i in &sample_idx {
                if col[i as usize] < thr { left.push(i); } else { right.push(i); }
            }
            // Degenerate guard: if split didn't separate, make a leaf.
            if left.is_empty() || right.is_empty() {
                let yv: Vec<f64> = sample_idx.iter().map(|&i| y[i as usize]).collect();
                return arena.alloc(NodeData::Leaf(yv));
            }
            let nid = arena.alloc(NodeData::Leaf(vec![]));
            let lid = fit_tree_cm(arena, cm, y, left, depth + 1, max_depth, min_ss, crit, require_decrease);
            let rid = fit_tree_cm(arena, cm, y, right, depth + 1, max_depth, min_ss, crit, require_decrease);
            arena.nodes[nid] = NodeData::Internal { feat, thr, left: lid, right: rid };
            nid
        }
    }
}

/// Route query rows through the tree. `rows` are (query_index, &feature_row) pairs;
/// query_index is the caller-supplied label, feature_row is that query point's features.
/// At each leaf we return the labels that arrived there paired with the leaf's stored
/// training y-values — matching DisTreebution's get_values_leaf contract.
fn collect_leaves_rows(
    arena: &Arena,
    node: usize,
    rows: Vec<(usize, usize)>, // (label, position-in-X)
    x: &[Vec<f64>],
) -> Vec<(Vec<usize>, Vec<f64>)> {
    match &arena.nodes[node] {
        NodeData::Leaf(yv) => {
            let labels: Vec<usize> = rows.iter().map(|&(lab, _)| lab).collect();
            vec![(labels, yv.clone())]
        }
        NodeData::Internal { feat, thr, left, right } => {
            let (f, t, l, r) = (*feat, *thr, *left, *right);
            let mut li = Vec::new();
            let mut ri = Vec::new();
            for (lab, posx) in rows {
                if x[posx][f] < t { li.push((lab, posx)); } else { ri.push((lab, posx)); }
            }
            let mut res = Vec::new();
            if !li.is_empty() { res.extend(collect_leaves_rows(arena, l, li, x)); }
            if !ri.is_empty() { res.extend(collect_leaves_rows(arena, r, ri, x)); }
            res
        }
    }
}

/// Entry point matching Python: `x` is the query matrix, `indexes[k]` labels row k of x.
fn collect_leaves(arena: &Arena, node: usize, x: &[Vec<f64>], idxs: Vec<usize>) -> Vec<(Vec<usize>, Vec<f64>)> {
    // Pair each provided label with its row position in x. If labels and rows are the
    // same length we map label k -> row k (Python's contract). If x is empty (caller
    // passed no query matrix) we fall back to treating idxs as row positions is not
    // possible, so require x to have at least as many rows as idxs.
    let rows: Vec<(usize, usize)> = idxs.iter().enumerate().map(|(k, &lab)| (lab, k)).collect();
    collect_leaves_rows(arena, node, rows, x)
}

/// Shared driver for all three tree flavours except quantile (which keeps the
/// original closure path). Builds column-major, runs fit_tree_cm.
fn fit_driver(
    x: &[Vec<f64>],
    y: &[f64],
    max_depth: Option<usize>,
    min_ss: usize,
    crit: Criterion,
    require_decrease: bool,
) -> (Arena, usize) {
    let cm = ColMajor::from_rows(x);
    let mut arena = Arena::new();
    let sample_idx: Vec<u32> = (0..cm.n_rows as u32).collect();
    let root = fit_tree_cm(&mut arena, &cm, y, sample_idx, 0, max_depth, min_ss, crit, require_decrease);
    (arena, root)
}

// ─────────────────────────────────────────────────────────────────────────────
// Python tree classes
// ─────────────────────────────────────────────────────────────────────────────
#[pyclass]
pub struct RegressionTreeQuadratic {
    max_depth: Option<usize>, min_ss: usize, loo: bool,
    arena: Arena, root: Option<usize>, train_x: Vec<Vec<f64>>,
}
#[pymethods]
impl RegressionTreeQuadratic {
    #[new]
    #[pyo3(signature = (max_depth=None, min_samples_split=2, loo=false))]
    pub fn new(max_depth: Option<usize>, min_samples_split: usize, loo: bool) -> Self {
        Self { max_depth, min_ss: min_samples_split, loo, arena: Arena::new(), root: None, train_x: Vec::new() }
    }
    pub fn fit(&mut self, py: Python<'_>, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let (max_depth, min_ss, loo) = (self.max_depth, self.min_ss, self.loo);
        let (arena, root) = py.allow_threads(|| {
            fit_driver(&x, &y, max_depth, min_ss, Criterion::Quadratic { loo }, loo)
        });
        self.train_x = x; self.arena = arena; self.root = Some(root);
        Ok(())
    }
    pub fn get_values_leaf(&self, x: Vec<Vec<f64>>, indexes: Vec<usize>) -> PyResult<Vec<(Vec<usize>, Vec<f64>)>> {
        let r=self.root.ok_or_else(|| PyValueError::new_err("Tree not fitted"))?;
        // `x` is the query matrix; row k corresponds to indexes[k] (Python contract).
        // If the caller passes an empty x but non-empty indexes, fall back to train_x
        // so that get_values_leaf(train_X, arange(n)) still works.
        let xref: &[Vec<f64>] = if x.is_empty() { &self.train_x } else { &x };
        Ok(collect_leaves(&self.arena, r, xref, indexes))
    }
}

#[pyclass]
pub struct RegressionTreeCRPS {
    max_depth: Option<usize>, min_ss: usize, loo: bool,
    arena: Arena, root: Option<usize>, train_x: Vec<Vec<f64>>,
}
#[pymethods]
impl RegressionTreeCRPS {
    #[new]
    #[pyo3(signature = (max_depth=None, min_samples_split=2, loo=false))]
    pub fn new(max_depth: Option<usize>, min_samples_split: usize, loo: bool) -> Self {
        Self { max_depth, min_ss: min_samples_split, loo, arena: Arena::new(), root: None, train_x: Vec::new() }
    }
    pub fn fit(&mut self, py: Python<'_>, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let (max_depth, min_ss, loo) = (self.max_depth, self.min_ss, self.loo);
        let (arena, root) = py.allow_threads(|| {
            // CRPS keeps require_decrease=true to match the original semantics.
            fit_driver(&x, &y, max_depth, min_ss, Criterion::Crps { loo }, true)
        });
        self.train_x = x; self.arena = arena; self.root = Some(root);
        Ok(())
    }
    pub fn get_values_leaf(&self, x: Vec<Vec<f64>>, indexes: Vec<usize>) -> PyResult<Vec<(Vec<usize>, Vec<f64>)>> {
        let r=self.root.ok_or_else(|| PyValueError::new_err("Tree not fitted"))?;
        // `x` is the query matrix; row k corresponds to indexes[k] (Python contract).
        // If the caller passes an empty x but non-empty indexes, fall back to train_x
        // so that get_values_leaf(train_X, arange(n)) still works.
        let xref: &[Vec<f64>] = if x.is_empty() { &self.train_x } else { &x };
        Ok(collect_leaves(&self.arena, r, xref, indexes))
    }
}

#[pyclass]
pub struct RegressionTreeQuantile {
    max_depth: Option<usize>, min_ss: usize, loo: bool, quantiles: Vec<f64>,
    arena: Arena, root: Option<usize>, train_x: Vec<Vec<f64>>,
}
#[pymethods]
impl RegressionTreeQuantile {
    #[new]
    #[pyo3(signature = (quantiles, max_depth=None, min_samples_split=2, loo=false))]
    pub fn new(quantiles: Vec<f64>, max_depth: Option<usize>, min_samples_split: usize, loo: bool) -> Self {
        Self { max_depth, min_ss: min_samples_split, loo, quantiles, arena: Arena::new(), root: None, train_x: Vec::new() }
    }
    pub fn fit(&mut self, py: Python<'_>, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let (max_depth, min_ss, loo) = (self.max_depth, self.min_ss, self.loo);
        let qs = self.quantiles.clone();
        let (arena, root) = py.allow_threads(|| {
            fit_quantile_driver(&x, &y, max_depth, min_ss, &qs, loo)
        });
        self.train_x = x; self.arena = arena; self.root = Some(root);
        Ok(())
    }
    pub fn get_values_leaf(&self, x: Vec<Vec<f64>>, indexes: Vec<usize>) -> PyResult<Vec<(Vec<usize>, Vec<f64>)>> {
        let r=self.root.ok_or_else(|| PyValueError::new_err("Tree not fitted"))?;
        // `x` is the query matrix; row k corresponds to indexes[k] (Python contract).
        // If the caller passes an empty x but non-empty indexes, fall back to train_x
        // so that get_values_leaf(train_X, arange(n)) still works.
        let xref: &[Vec<f64>] = if x.is_empty() { &self.train_x } else { &x };
        Ok(collect_leaves(&self.arena, r, xref, indexes))
    }
}

// Quantile trees keep the (heavier) multi-quantile entropy; we still benefit from
// column-major storage + O(1) left-count + parallel features.
fn fit_quantile_driver(
    x: &[Vec<f64>],
    y: &[f64],
    max_depth: Option<usize>,
    min_ss: usize,
    quantiles: &[f64],
    loo: bool,
) -> (Arena, usize) {
    let cm = ColMajor::from_rows(x);
    let mut arena = Arena::new();
    let sample_idx: Vec<u32> = (0..cm.n_rows as u32).collect();
    let root = fit_tree_quantile(&mut arena, &cm, y, sample_idx, 0, max_depth, min_ss, quantiles, loo);
    (arena, root)
}

fn best_split_for_feature_quantile(
    col: &[f64],
    sample_idx: &[u32],
    y: &[f64],
    quantiles: &[f64],
    min_ss: usize,
    loo: bool,
) -> Option<(f64, f64)> {
    let ns = sample_idx.len();
    if ns <= 1 { return None; }
    let mut order: Vec<u32> = sample_idx.to_vec();
    order.sort_unstable_by(|&a, &b| {
        col[a as usize].total_cmp(&col[b as usize])
    });
    let order_us: Vec<usize> = order.iter().map(|&a| a as usize).collect();
    let eu = entropies_multi_quantiles(order_us.clone(), y.to_vec(), quantiles.to_vec(), loo);
    let mut rev = order_us.clone(); rev.reverse();
    let ed = entropies_multi_quantiles(rev, y.to_vec(), quantiles.to_vec(), loo);
    let len = ns + 1;
    let global_score = (len as f64 - 1.0) * eu[len - 1];
    let mut best: Option<(f64, f64)> = None;
    for i in 1..len - 1 {
        if i < min_ss || ns - i < min_ss { continue; }
        let xa = col[order[i - 1] as usize];
        let xb = col[order[i] as usize];
        if (xa - xb).abs() <= 1e-5 { continue; }
        let score = (i as f64) * eu[i] + ((len - 1 - i) as f64) * ed[len - 1 - i];
        if !(score < global_score) { continue; }
        match best {
            None => best = Some((score, (xa + xb) / 2.0)),
            Some((bs, _)) if score < bs => best = Some((score, (xa + xb) / 2.0)),
            _ => {}
        }
    }
    best
}

fn fit_tree_quantile(
    arena: &mut Arena,
    cm: &ColMajor,
    y: &[f64],
    sample_idx: Vec<u32>,
    depth: usize,
    max_depth: Option<usize>,
    min_ss: usize,
    quantiles: &[f64],
    loo: bool,
) -> usize {
    let ns = sample_idx.len();
    let stop = max_depth.map(|md| depth >= md).unwrap_or(false) || ns < min_ss || ns <= 1;
    if stop {
        let yv: Vec<f64> = sample_idx.iter().map(|&i| y[i as usize]).collect();
        return arena.alloc(NodeData::Leaf(yv));
    }
    const PAR_THRESHOLD: usize = 256;
    let nf = cm.n_cols;
    let best: Option<(f64, f64, usize)> = if ns >= PAR_THRESHOLD {
        (0..nf).into_par_iter()
            .filter_map(|feat| best_split_for_feature_quantile(&cm.cols[feat], &sample_idx, y, quantiles, min_ss, loo)
                .map(|(s, t)| (s, t, feat)))
            .reduce_with(|a, b| if a.0 <= b.0 { a } else { b })
    } else {
        let mut acc: Option<(f64, f64, usize)> = None;
        for feat in 0..nf {
            if let Some((s, t)) = best_split_for_feature_quantile(&cm.cols[feat], &sample_idx, y, quantiles, min_ss, loo) {
                match acc { None => acc = Some((s, t, feat)), Some(a) if s < a.0 => acc = Some((s, t, feat)), _ => {} }
            }
        }
        acc
    };
    match best {
        None => {
            let yv: Vec<f64> = sample_idx.iter().map(|&i| y[i as usize]).collect();
            arena.alloc(NodeData::Leaf(yv))
        }
        Some((_s, thr, feat)) => {
            let col = &cm.cols[feat];
            let mut left = Vec::new(); let mut right = Vec::new();
            for &i in &sample_idx {
                if col[i as usize] < thr { left.push(i); } else { right.push(i); }
            }
            if left.is_empty() || right.is_empty() {
                let yv: Vec<f64> = sample_idx.iter().map(|&i| y[i as usize]).collect();
                return arena.alloc(NodeData::Leaf(yv));
            }
            let nid = arena.alloc(NodeData::Leaf(vec![]));
            let lid = fit_tree_quantile(arena, cm, y, left, depth + 1, max_depth, min_ss, quantiles, loo);
            let rid = fit_tree_quantile(arena, cm, y, right, depth + 1, max_depth, min_ss, quantiles, loo);
            arena.nodes[nid] = NodeData::Internal { feat, thr, left: lid, right: rid };
            nid
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module
// ─────────────────────────────────────────────────────────────────────────────
#[pymodule]
fn distreebu_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FenwickTree>()?;
    m.add_class::<MinMaxHeap>()?;
    m.add_class::<RegressionTreeQuadratic>()?;
    m.add_class::<RegressionTreeCRPS>()?;
    m.add_class::<RegressionTreeQuantile>()?;
    m.add_function(wrap_pyfunction!(entropies_quadratic, m)?)?;
    m.add_function(wrap_pyfunction!(entropies_crps, m)?)?;
    m.add_function(wrap_pyfunction!(entropies_multi_quantiles, m)?)?;
    Ok(())
}
