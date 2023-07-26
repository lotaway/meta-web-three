mod utils;
mod matcher;

fn test() {
    matcher::test_color_matcher();
    utils::get_config_file(".\\config\\public.json");
}

fn main() {
    test();
}