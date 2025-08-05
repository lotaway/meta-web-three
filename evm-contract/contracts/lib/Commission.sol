// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

library CommissionLib {
    uint8 public constant MAX_LEVEL = 7; // 最大层级数

    enum CommissionBaseType {
        ON_BUYER,
        ON_LAST_ONE,
    }
}