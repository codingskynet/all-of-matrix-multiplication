use std::{
    alloc::{alloc, alloc_zeroed, Layout},
    cmp::{max, min},
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub},
    ptr::{drop_in_place, NonNull},
    slice,
};

use rand::{distributions::uniform::SampleUniform, thread_rng, Rng};

pub struct Matrix2D<T> {
    raw: NonNull<T>,
    row: usize,
    column: usize,
}

impl<T> Drop for Matrix2D<T> {
    fn drop(&mut self) {
        unsafe { drop_in_place(self.raw.as_ptr()) }
    }
}

impl<T: Debug> Debug for Matrix2D<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let matrix = "[".to_string()
            + &(0..self.row)
                .map(|r| {
                    "[".to_string()
                        + &self
                            .row(r)
                            .iter()
                            .map(|t| format!("{:?}", t))
                            .collect::<Vec<_>>()
                            .join(", ")
                        + "]"
                })
                .collect::<Vec<_>>()
                .join(", ")
            + "]";
        f.debug_struct("Matrix2D")
            .field("raw", &matrix)
            .field("row", &self.row)
            .field("column", &self.column)
            .finish()
    }
}

impl<T: Display> Display for Matrix2D<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let matrix = "[".to_string()
            + &(0..self.row)
                .map(|r| {
                    self.row(r)
                        .iter()
                        .map(|t| format!("{:>12}", t))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .collect::<Vec<_>>()
                .join(",\n ")
            + "]";

        write!(f, "\n{}, ({}, {})", matrix, self.row, self.column)
    }
}

enum InitType {
    UnInit,
    Zeroed,
}

impl<T> Matrix2D<T> {
    fn alloc(row: usize, column: usize, init: InitType) -> Self {
        if row * column == 0 {
            panic!("Cannot create zero matrix");
        }

        let layout = match Layout::array::<T>(row * column) {
            Ok(layout) => layout,
            Err(_) => panic!(
                "Cannot initialize the size of matrix: ({}, {})",
                row, column
            ),
        };

        let raw = unsafe {
            match init {
                InitType::UnInit => NonNull::new(alloc(layout) as *mut T).unwrap(),
                InitType::Zeroed => NonNull::new(alloc_zeroed(layout) as *mut T).unwrap(),
            }
        };

        Self { raw, row, column }
    }

    pub fn new(row: usize, column: usize) -> Self {
        Self::alloc(row, column, InitType::UnInit)
    }

    pub fn new_zeroed(row: usize, column: usize) -> Self {
        Self::alloc(row, column, InitType::Zeroed)
    }

    pub fn from(array: Vec<T>, row: usize, column: usize) -> Self {
        assert_eq!(array.len(), row * column);

        let mut uninit = Self::new(row, column);

        for (element, value) in uninit.flatten_mut().iter_mut().zip(array) {
            *element = value;
        }

        uninit
    }

    pub fn new_uniform_rand(row: usize, column: usize, from: T, to: T) -> Self
    where
        T: Copy + PartialOrd + SampleUniform,
    {
        let mut uninit = Self::new(row, column);

        let mut rng = thread_rng();
        for r in 0..uninit.row {
            for element in uninit.row_mut(r) {
                *element = rng.gen_range(from..to);
            }
        }

        uninit
    }

    pub fn flatten(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.raw.as_ptr(), self.row * self.column) }
    }

    pub fn flatten_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.raw.as_ptr(), self.row * self.column) }
    }

    pub fn row(&self, row: usize) -> &[T] {
        unsafe { slice::from_raw_parts(self.raw.as_ptr().add(row * self.column), self.column) }
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.raw.as_ptr().add(row * self.column), self.column) }
    }

    pub fn add(a: &Self, b: &Self) -> Self
    where
        T: Copy + Add<Output = T>,
    {
        debug_assert_eq!(a.row, b.row);
        debug_assert_eq!(a.column, b.column);

        let mut c = Matrix2D::new(a.row, a.column);

        for r in 0..a.row {
            for (i, (a, b)) in a.row(r).iter().zip(b.row(r)).enumerate() {
                c[(r, i)] = *a + *b;
            }
        }

        c
    }

    pub fn sub(a: &Self, b: &Self) -> Self
    where
        T: Copy + Sub<Output = T>,
    {
        debug_assert_eq!(a.row, b.row);
        debug_assert_eq!(a.column, b.column);

        let mut c = Matrix2D::new(a.row, a.column);

        for r in 0..a.row {
            for (i, (a, b)) in a.row(r).iter().zip(b.row(r)).enumerate() {
                c[(r, i)] = *a - *b;
            }
        }

        c
    }

    pub fn norm_inf(&self) -> T
    where
        T: Copy + PartialOrd + Neg<Output = T> + Sum,
    {
        fn abs<T: Copy + PartialOrd + Neg<Output = T>>(e: T) -> T {
            if e > -e {
                e
            } else {
                -e
            }
        }

        let mut result = self.row(0).iter().map(|e| abs(*e)).sum::<T>();

        for r in 1..self.row {
            let sum = self.row(r).iter().map(|e| abs(*e)).sum::<T>();

            if result < sum {
                result = sum;
            }
        }

        result
    }

    pub fn naive_multiply(a: &Self, b: &Self) -> Self
    where
        T: Copy + Mul<Output = T> + AddAssign,
    {
        assert_eq!(a.column, b.row);

        let mut c = Self::new_zeroed(a.row, b.column);

        for i in 0..a.row {
            for j in 0..b.column {
                for k in 0..b.row {
                    c[(i, j)] += a[(i, k)] * b[(k, j)];
                }
            }
        }

        c
    }

    pub fn naive_multiply_optimized(a: &Self, b: &Self) -> Self
    where
        T: Copy + Mul<Output = T> + AddAssign,
    {
        assert_eq!(a.column, b.row);

        let mut c = Self::new_zeroed(a.row, b.column);

        for k in 0..b.row {
            for i in 0..a.row {
                for j in 0..b.column {
                    c[(i, j)] += a[(i, k)] * b[(k, j)];
                }
            }
        }

        c
    }

    pub fn naive_multiply_optimized_block(a: &Self, b: &Self) -> Self
    where
        T: Copy + Mul<Output = T> + AddAssign,
    {
        assert_eq!(a.column, b.row);

        let mut c = Self::new_zeroed(a.row, b.column);

        let block = 8;

        for k_block in (0..b.row).step_by(block) {
            for i_block in (0..a.row).step_by(block) {
                for j_block in (0..b.column).step_by(block) {
                    for k in k_block..min(k_block + block, b.row) {
                        for i in i_block..min(i_block + block, a.row) {
                            for j in j_block..min(j_block + block, b.column) {
                                c[(i, j)] += a[(i, k)] * b[(k, j)];
                            }
                        }
                    }
                }
            }
        }

        c
    }

    pub fn strassen_multiply(a: &Self, b: &Self) -> Self
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign,
    {
        assert_eq!(a.column, b.row);
        // now, odd matrix is not supported.
        assert!(a.row % 2 == 0 && a.column % 2 == 0 && b.row % 2 == 0 && b.column % 2 == 0);

        // In the small case, strassen is inefficient. Thus, use naive
        if max(a.row * a.column, b.row * b.column) <= 10000 {
            Matrix2D::naive_multiply_optimized_block(a, b)
        } else {
            let quarter_a = a.quarterize();
            let quarter_b = b.quarterize();

            // from https://en.wikipedia.org/wiki/Strassen_algorithm
            let m_1 = Matrix2D::strassen_multiply(
                &Matrix2D::add(&quarter_a[0], &quarter_a[3]),
                &Matrix2D::add(&quarter_b[0], &quarter_b[3]),
            );
            let m_2 = Matrix2D::strassen_multiply(
                &Matrix2D::add(&quarter_a[2], &quarter_a[3]),
                &quarter_b[0],
            );
            let m_3 = Matrix2D::strassen_multiply(
                &quarter_a[0],
                &Matrix2D::sub(&quarter_b[1], &quarter_b[3]),
            );
            let m_4 = Matrix2D::strassen_multiply(
                &quarter_a[3],
                &Matrix2D::sub(&quarter_b[2], &quarter_b[0]),
            );
            let m_5 = Matrix2D::strassen_multiply(
                &Matrix2D::add(&quarter_a[0], &quarter_a[1]),
                &quarter_b[3],
            );
            let m_6 = Matrix2D::strassen_multiply(
                &Matrix2D::sub(&quarter_a[2], &quarter_a[0]),
                &Matrix2D::add(&quarter_b[0], &quarter_b[1]),
            );
            let m_7 = Matrix2D::strassen_multiply(
                &Matrix2D::sub(&quarter_a[1], &quarter_a[3]),
                &Matrix2D::add(&quarter_b[2], &quarter_b[3]),
            );

            let c_1 = Matrix2D::add(&Matrix2D::sub(&m_1, &m_5), &Matrix2D::add(&m_4, &m_7));
            let c_2 = Matrix2D::add(&m_3, &m_5);
            let c_3 = Matrix2D::add(&m_2, &m_4);
            let c_4 = Matrix2D::add(&Matrix2D::sub(&m_1, &m_2), &Matrix2D::add(&m_3, &m_6));

            Matrix2D::dequarterize(&[c_1, c_2, c_3, c_4])
        }
    }

    pub fn modified_strassen_multiply(a: &Self, b: &Self) -> Self
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + AddAssign,
    {
        assert_eq!(a.column, b.row);
        // now, odd matrix is not supported.
        assert!(a.row % 2 == 0 && a.column % 2 == 0 && b.row % 2 == 0 && b.column % 2 == 0);

        // In the small case, strassen is inefficient. Thus, use naive
        if max(a.row * a.column, b.row * b.column) <= 10000 {
            Matrix2D::naive_multiply_optimized_block(a, b)
        } else {
            let quarter_a = a.quarterize();
            let quarter_b = b.quarterize();

            // from https://ko.wikipedia.org/wiki/%EC%8A%88%ED%8A%B8%EB%9D%BC%EC%84%BC_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98#%EB%B3%80%ED%98%95%EB%90%9C_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
            let s_1 = Matrix2D::add(&quarter_a[2], &quarter_a[3]);
            let s_2 = Matrix2D::sub(&s_1, &quarter_a[0]);
            let s_3 = Matrix2D::sub(&quarter_b[1], &quarter_b[0]);
            let s_4 = Matrix2D::sub(&quarter_b[3], &s_3);

            let m_1 = Matrix2D::modified_strassen_multiply(&s_2, &s_4);
            let m_2 = Matrix2D::modified_strassen_multiply(&quarter_a[0], &quarter_b[0]);
            let m_3 = Matrix2D::modified_strassen_multiply(&quarter_a[1], &quarter_b[2]);
            let m_4 = Matrix2D::modified_strassen_multiply(
                &Matrix2D::sub(&quarter_a[0], &quarter_a[2]),
                &Matrix2D::sub(&quarter_b[3], &quarter_b[1]),
            );
            let m_5 = Matrix2D::modified_strassen_multiply(&s_1, &s_3);
            let m_6 = Matrix2D::modified_strassen_multiply(
                &Matrix2D::sub(&quarter_a[1], &s_2),
                &quarter_b[3],
            );
            let m_7 = Matrix2D::modified_strassen_multiply(
                &quarter_a[3],
                &Matrix2D::sub(&s_4, &quarter_b[2]),
            );

            let t_1 = Matrix2D::add(&m_1, &m_2);
            let t_2 = Matrix2D::add(&t_1, &m_4);

            let c_1 = Matrix2D::add(&m_2, &m_3);
            let c_2 = Matrix2D::add(&Matrix2D::add(&t_1, &m_5), &m_6);
            let c_3 = Matrix2D::sub(&t_2, &m_7);
            let c_4 = Matrix2D::add(&t_2, &m_5);

            Matrix2D::dequarterize(&[c_1, c_2, c_3, c_4])
        }
    }

    pub fn quarterize(&self) -> [Self; 4]
    where
        T: Copy,
    {
        assert_eq!(self.row % 2, 0);
        assert_eq!(self.column % 2, 0);

        // matrix like:
        // [ [0], [1]
        //   [2], [3] ]
        let mut result = [
            Matrix2D::new(self.row / 2, self.column / 2),
            Matrix2D::new(self.row / 2, self.column / 2),
            Matrix2D::new(self.row / 2, self.column / 2),
            Matrix2D::new(self.row / 2, self.column / 2),
        ];

        // for result[0], result[1]
        for r in 0..(self.row / 2) {
            for (i, e) in self.row(r).iter().enumerate() {
                if i < self.column / 2 {
                    result[0][(r, i)] = *e;
                } else {
                    result[1][(r, i - self.column / 2)] = *e;
                }
            }
        }

        // for result[2], result[3]
        for r in (self.row / 2)..self.row {
            for (i, e) in self.row(r).iter().enumerate() {
                if i < self.column / 2 {
                    result[2][(r - self.row / 2, i)] = *e;
                } else {
                    result[3][(r - self.row / 2, i - self.column / 2)] = *e;
                }
            }
        }

        result
    }

    pub fn dequarterize(a: &[Self; 4]) -> Self
    where
        T: Copy,
    {
        let mut result = Matrix2D::new(a[0].row * 2, a[0].column * 2);

        for r in 0..a[0].row {
            for (i, e) in a[0].row(r).iter().enumerate() {
                result[(r, i)] = *e;
            }
        }

        for r in 0..a[1].row {
            for (i, e) in a[1].row(r).iter().enumerate() {
                result[(r, i + a[0].column)] = *e;
            }
        }

        for r in 0..a[2].row {
            for (i, e) in a[2].row(r).iter().enumerate() {
                result[(a[0].row + r, i)] = *e;
            }
        }

        for r in 0..a[3].row {
            for (i, e) in a[3].row(r).iter().enumerate() {
                result[(a[1].row + r, a[2].column + i)] = *e;
            }
        }

        result
    }
}

impl<T> Index<(usize, usize)> for Matrix2D<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        unsafe { &*self.raw.as_ptr().add(index.0 * self.column + index.1) }
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix2D<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        unsafe { &mut *self.raw.as_ptr().add(index.0 * self.column + index.1) }
    }
}

impl<T: PartialEq> PartialEq for Matrix2D<T> {
    fn eq(&self, other: &Self) -> bool {
        if !(self.row == other.row && self.column == self.column) {
            return false;
        }

        for (s, o) in self.flatten().iter().zip(other.flatten()) {
            if s != o {
                return false;
            }
        }

        true
    }
}
