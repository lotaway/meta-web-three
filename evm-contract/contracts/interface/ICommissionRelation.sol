// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ICommissionRelation {
    function isValidDistributor(address distributor) external view returns (bool);
    function getUplines(address account) external view returns (address[] memory);
    function setUpline(address account, address upline) external;
    function getDownlineCount(address account) external view returns (uint256);
}