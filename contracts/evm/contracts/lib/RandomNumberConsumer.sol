// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@chainlink/contracts/src/v0.8/vrf/VRFConsumerBase.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract RandomNumberConsumer is VRFConsumerBase, Ownable {
    bytes32 internal keyHash;
    uint256 internal fee;
    uint256 public randomResult;
    
    constructor(address _vrfCoordinator, address _link, bytes32 _keyHash) 
        VRFConsumerBase(
            _vrfCoordinator,
            _link
        )
        Ownable(msg.sender)
    {
        keyHash = _keyHash;
        fee = 0.1 * 10 ** 18; // 0.1 LINK
    }

    function setFee(uint256 _fee) public onlyOwner {
        fee = _fee;
    }
    
    function initRandomNumber() public onlyOwner returns (bytes32 requestId) {
        require(LINK.balanceOf(address(this)) >= fee, "Not enough LINK");
        return requestRandomness(keyHash, fee);
    }

    function reflashRandomNumber() public returns (bytes32 requestId) {
        require(LINK.balanceOf(address(msg.sender)) >= fee, "Not enough LINK");
        return requestRandomness(keyHash, fee);
    }
    
    /// @dev Callback function used by VRF Coordinator
    function fulfillRandomness(bytes32 requestId, uint256 randomness) internal override {
        randomResult = randomness;
        // use randomness to do something
        uint256 percent = (randomness * 100) / type(uint256).max;
    }
}