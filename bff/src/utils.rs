pub struct UniqueIterators<I>
where
    I: Iterator,
{
    iter: I,
    seen: std::collections::HashSet<I::Item>,
}

pub trait IteratorExt: Iterator {
    fn unique(self) -> UniqueIterators<Self>
    where
        Self: Sized,
        Self::Item: std::cmp::Eq + std::hash::Hash + Clone,
    {
        UniqueIterators {
            iter: self,
            seen: std::collections::HashSet::new(),
        }
    }
}

impl<I> Iterator for UniqueIterators<I>
where
    I: Iterator,
    I::Item: std::cmp::Eq + std::cmp::PartialEq + std::hash::Hash + Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> std::option::Option<Self::Item> {
        self.iter.find(|item| self.seen.insert(item.clone()))
    }
}

impl<I> IteratorExt for I where I: Iterator {}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_iterator() {
        let vec = vec![1, 2, 2, 3, 4, 4, 5];
        let unique_vec: Vec<_> = vec.into_iter().unique().collect();
        assert_eq!(unique_vec, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unique_iterator_empty() {
        let vec: Vec<i32> = vec![];
        let unique_vec = vec.into_iter().unique().collect::<Vec<i32>>();
        assert_eq!(unique_vec, vec![] as Vec<i32>);
    }

    #[test]
    fn test_unique_iterator_all_unique() {
        let vec = vec![1, 2, 3, 4, 5];
        let unique_vec: Vec<_> = vec.into_iter().unique().collect();
        assert_eq!(unique_vec, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unique_iterator_all_same() {
        let vec = vec![1, 1, 1, 1, 1];
        let unique_vec: Vec<_> = vec.into_iter().unique().collect();
        assert_eq!(unique_vec, vec![1]);
    }

    #[test]
    fn test_unique_iterator_string() {
        let vec = vec!["a", "b", "b", "c", "d", "d", "e"];
        let unique_vec: Vec<_> = vec.into_iter().unique().collect();
        assert_eq!(unique_vec, vec!["a", "b", "c", "d", "e"]);
    }
}