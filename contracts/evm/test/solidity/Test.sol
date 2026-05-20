// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface Vm {
    function prank(address) external;
    function startPrank(address) external;
    function stopPrank() external;
    function warp(uint256) external;
    function expectRevert(bytes calldata) external;
    function assume(bool) external;
    function targetContract(address) external;
}

contract Test {
    Vm internal constant vm =
        Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    function assertTrue(bool condition, string memory message) internal pure {
        require(condition, message);
    }

    function assertTrue(bool condition) internal pure {
        require(condition, "assertTrue failed");
    }

    function assertEq(uint256 a, uint256 b) internal pure {
        require(a == b, "assertEq failed");
    }

    function assertEq(address a, address b) internal pure {
        require(a == b, "assertEq failed");
    }
}
