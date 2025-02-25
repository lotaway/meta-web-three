// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface ICommissionToken {
    function mint(address to, uint256 amount) external returns (bool);
    function burn(address from, uint256 amount) external returns (bool);
}