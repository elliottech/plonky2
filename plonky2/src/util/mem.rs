// alloc memory for Vec<F>, where every element is 0. (a lot) faster than vec![F::ZERO; len]
pub unsafe fn vec_zeroed<F>(len: usize) -> Vec<F> {
    let elem_size = std::mem::size_of::<F>();
    debug_assert!(elem_size != 0, "ZST not supported by this helper");

    // Layout for len elements
    let layout = std::alloc::Layout::array::<F>(len).expect("layout overflow");

    // Allocate zeroed memory
    let ptr = std::alloc::alloc_zeroed(layout) as *mut F;
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }

    // Take ownership as a Vec<F>
    Vec::from_raw_parts(ptr, len, len)
}
