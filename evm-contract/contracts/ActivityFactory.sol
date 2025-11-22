// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "./Activity.sol";

contract ActivityFactory is Ownable {
    address public metaThreeCoin;

    event ActivityCreated(
        address indexed activityAddress,
        uint256 startTime,
        uint256 endTime,
        uint256 entryFee
    );

    Activity[] public activities;

    constructor(address _metaThreeCoin) Ownable(msg.sender) {
        metaThreeCoin = _metaThreeCoin;
    }

    function createActivity(
        uint256 startTime,
        uint256 endTime,
        uint256[3] memory rewardPercentages,
        uint256 entryFee
    ) external onlyOwner {
        Activity activity = new Activity(
            startTime,
            endTime,
            rewardPercentages,
            entryFee,
            metaThreeCoin,
            msg.sender // Admin
        );

        activities.push(activity);

        emit ActivityCreated(address(activity), startTime, endTime, entryFee);
    }

    function getActivities() external view returns (Activity[] memory) {
        return activities;
    }
}
