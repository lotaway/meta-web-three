// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

enum GoodsStatus {
    OffShelf,
    OnShelf,
    SoldOut
}

// 商品规格结构
struct Specification {
    string key;
    string value;
}

// 商品结构
struct Good {
    string name;
    Specification[] specifications;
}