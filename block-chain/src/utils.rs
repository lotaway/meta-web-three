use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

pub fn show_info(message: &str) {
    println!("show info: {}", message);
}

// type ListNodeValue<Rhs> = dyn PartialEq<Rhs> + Clone;
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct SingleLinkedList<T: PartialEq + Clone> {
    pub val: T,
    pub next: Option<NonNull<SingleLinkedList<T>>>,
}

impl<T: PartialEq + Clone> SingleLinkedList<T> {
    #[inline]
    pub fn new(val: T) -> Self {
        SingleLinkedList {
            next: None,
            val,
        }
    }

    pub unsafe fn del_dup(&mut self) {
        let mut node = self;
        let mut cur_val = node.val.clone();
        while let Option::Some(next) = node.next.take() { // 拿走
            if next.as_ref().val == cur_val {
                node.next = next.as_ref().next;
            } else {
                cur_val = next.as_ref().val.clone();
                node.next = Option::Some(next); // 放回
                node = node.next.unwrap().as_mut();
            }
        }
    }
}

impl<T: PartialEq + Clone> From<&[T]> for SingleLinkedList<T> {
    fn from(list: &[T]) -> Self {
        let mut head = Option::None;
        for val in list.iter().rev() {
            let new_node = Box::new(SingleLinkedList {
                val: val.clone(),
                next: head,
            });
            head = NonNull::new(Box::into_raw(new_node));
        }
        // Safety: We know that head is not null because we just created it.
        let head = unsafe { *Box::from_raw(head.unwrap().as_ptr()) };
        head
    }
}

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode {
            next: Option::None,
            val,
        }
    }
}

pub fn delete_duplicates(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    if head.is_none() {
        return head;
    }
    let mut head = head;
    let mut node = head.as_mut().unwrap();
    let mut cur = node.val;
    while let Option::Some(next) = node.next.take() {
        if next.val == cur {
            node.next = next.next;
        } else {
            cur = next.val;
            node.next = Option::Some(next);
            node = node.next.as_mut().unwrap();
        }
    }
    head
}

pub fn find_sub_str_index(haystack: String, needle: String) -> i32 {
    let size = haystack.len();
    let c_size = needle.len();
    if size == 0 || c_size == 0 {
        return -1;
    }
    let mut index = 0;
    let mut it = needle.chars();
    let chars: Vec<char> = haystack.chars().collect();
    while index < size && c_size <= size - index {
        let mut c_count = 0;
        while let Option::Some(ch) = it.next() {
            if ch != chars[index + c_count] {
                it = needle.chars();
                break;
            }
            c_count = c_count + 1;
        }
        if c_count == c_size {
            return index as i32;
        }
        index = index + 1;
    }
    -1
}

pub fn get_config_file(file_name: &str) -> String {
    let prev_fix = "/config";
    let path = Path::new(prev_fix).join(file_name);
    let mut content = match get_file(&path) {
        Ok(_content) => _content,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create(file_name) {
                Ok(mut n_file) => {
                    let mut _content = String::new();
                    n_file.read_to_string(&mut _content);
                    _content
                }
                Err(err) => panic!("{}", err.to_string())
            },
            _ => panic!("{}", error.to_string())
        }
    };
    if content.is_empty() {
        std::io::stdin().read_line(&mut content).expect("error in get config file.");
    }
    content
}

fn get_file(path: &PathBuf) -> Result<String, std::io::Error> {
    let mut content = String::new();
    // let size = File::open(path).and_then(|mut file| file.read_to_string(&mut content))?;
    // Ok(content)
    File::open(path)?.read_to_string(&mut content).map(|_| content)
}

pub fn climb_stairs(n: i32) -> i32 {
    if n <= 2 {
        return n;
    }
    let mut sums = vec![0; n as usize + 1];
    sums[1] = 1;
    sums[2] = 2;
    for i in 3usize..n as usize + 1 {
        sums[i] = sums[i - 1] + sums[i - 2];
    }
    sums[n as usize]
}

#[cfg(test)]
pub mod utils_tests {
    use crate::utils::{climb_stairs, find_sub_str_index, ListNode, SingleLinkedList};
    use super::delete_duplicates;

    #[test]
    fn test_delete_duplicates_link_node() {
        /*{
            let is_none = None::<Box<SingleLinkedList<i8>>>;
            assert_eq!(SingleLinkedList::del_dup_from_option(is_none.clone()), is_none);
        }
        {
            let no_duplicate = Some(Box::new(SingleLinkedList { val: 0, next: None }));
            assert_eq!(SingleLinkedList::del_dup_from_option(no_duplicate.clone()), no_duplicate);
        }*/
        unsafe {
            let has_duplicate = SingleLinkedList::from(&[0, 1, 1, 2, 2, 2, 3, 4] as &[i32]);
            let mut clone = has_duplicate.clone();
            clone.del_dup();
            assert_eq!(clone, has_duplicate);
        }
    }

    fn build_list_from_slice(s: &[i32]) -> Option<Box<ListNode>> {
        if s.is_empty() {
            return Option::None;
        }
        let head = Box::new(ListNode {
            val: s.first().copied().unwrap(),
            next: build_list_from_slice(&s[1..]),
        });
        Option::Some(head)
    }

    #[test]
    fn test_delete_duplicates() {
        struct TestCase {
            name: &'static str,
            nums: &'static [i32],
            expect: &'static [i32],
        }

        vec![
            TestCase {
                name: "basic",
                nums: &[1, 1, 2, 3, 3],
                expect: &[1, 2, 3],
            },
        ].iter().for_each(|testcase| {
            let head = build_list_from_slice(testcase.nums);
            let actual = delete_duplicates(head);
            let expect = build_list_from_slice(testcase.expect);
            assert_eq!(expect, actual, "{} failed", testcase.name);
        });
    }

    #[test]
    fn test_find_sub_str_index() {
        let result = find_sub_str_index(String::from("sadbutsad"), String::from("sad"));
        assert_eq!(result, 0, "find sub str index failed 1");
        let result2 = find_sub_str_index(String::from("leetcode"), String::from("leeto"));
        assert_eq!(result2, -1, "find sub str index failed 2");
        let result3 = find_sub_str_index(String::from("mississippi"), String::from("issip"));
        assert_eq!(result3, 4, "find sub str index failed 3");
    }

    #[test]
    fn test_climb_stairs() {
        assert_eq!(climb_stairs(1), 1);
        assert_eq!(climb_stairs(2), 2);
        assert_eq!(climb_stairs(3), 3);
        assert_eq!(climb_stairs(5), 8);
        assert_eq!(climb_stairs(45), 1836311903);
    }
}