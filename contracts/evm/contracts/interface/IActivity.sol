// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IActivity {
    function participate() external;
    function claimReward(uint256 rank, bytes32[] calldata proof) external;
}
