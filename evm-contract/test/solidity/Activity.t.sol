// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Test.sol";
import "../../contracts/Activity.sol";
import "../../contracts/MetaThreeCoin.sol";

contract ActivityHandler is Test {
    Activity public activity;
    MetaThreeCoin public token;
    uint256 public participants;

    constructor(Activity _activity, MetaThreeCoin _token) {
        activity = _activity;
        token = _token;
    }

    function participate(address user) public {
        if (activity.hasParticipated(user)) {
            return;
        }
        uint256 fee = activity.entryFee();
        token.transfer(user, fee);
        vm.startPrank(user);
        token.approve(address(activity), fee);
        activity.participate();
        vm.stopPrank();
        participants += 1;
    }
}

contract ActivityTest is Test {
    MetaThreeCoin private token;
    Activity private activity;
    ActivityHandler private handler;

    function setUp() public {
        token = new MetaThreeCoin("MetaThreeCoin", "M3C");
        activity = new Activity(
            block.timestamp,
            block.timestamp + 1 days,
            [uint256(50), uint256(30), uint256(20)],
            100,
            address(token),
            address(this)
        );
        handler = new ActivityHandler(activity, token);
        vm.targetContract(address(handler));
    }

    function testParticipate() public {
        address user = address(0xBEEF);
        token.transfer(user, 200);
        vm.startPrank(user);
        token.approve(address(activity), 100);
        activity.participate();
        vm.stopPrank();

        assertEq(activity.totalPool(), 100);
        assertTrue(activity.hasParticipated(user));
    }

    function testFuzz_Participate(uint96 fee) public {
        vm.assume(fee > 0);
        Activity local = new Activity(
            block.timestamp,
            block.timestamp + 1 days,
            [uint256(50), uint256(30), uint256(20)],
            uint256(fee),
            address(token),
            address(this)
        );

        address user = address(0xCAFE);
        token.transfer(user, uint256(fee));
        vm.startPrank(user);
        token.approve(address(local), uint256(fee));
        local.participate();
        vm.stopPrank();

        assertEq(local.totalPool(), uint256(fee));
    }

    function invariant_TotalPoolEqualsEntryFeeTimesParticipants() public {
        assertEq(
            activity.totalPool(),
            activity.entryFee() * handler.participants()
        );
    }
}
