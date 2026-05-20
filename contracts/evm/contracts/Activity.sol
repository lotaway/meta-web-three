// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interface/IActivity.sol";

contract Activity is IActivity, Ownable, ReentrancyGuard {
    IERC20 public immutable metaThreeCoin;

    uint256 public startTime;
    uint256 public endTime;
    uint256[3] public rewardPercentages;
    uint256 public entryFee;
    uint256 public totalPool;
    bytes32 public merkleRoot;

    mapping(address => bool) public hasParticipated;
    mapping(address => bool) public hasClaimed;

    event Participated(address indexed participant);
    event RewardClaimed(address indexed winner, uint256 rank, uint256 amount);
    event MerkleRootSet(bytes32 merkleRoot);

    constructor(
        uint256 _startTime,
        uint256 _endTime,
        uint256[3] memory _rewardPercentages,
        uint256 _entryFee,
        address _metaThreeCoin,
        address _admin
    ) Ownable(_admin) {
        require(_startTime < _endTime, "Invalid time range");
        require(
            _rewardPercentages[0] +
                _rewardPercentages[1] +
                _rewardPercentages[2] <=
                100,
            "Invalid percentages"
        );

        startTime = _startTime;
        endTime = _endTime;
        rewardPercentages = _rewardPercentages;
        entryFee = _entryFee;
        metaThreeCoin = IERC20(_metaThreeCoin);
    }

    function participate() external override nonReentrant {
        require(block.timestamp >= startTime, "Activity not started");
        require(block.timestamp <= endTime, "Activity ended");
        require(!hasParticipated[msg.sender], "Already participated");

        require(
            metaThreeCoin.transferFrom(msg.sender, address(this), entryFee),
            "Transfer failed"
        );

        totalPool += entryFee;
        hasParticipated[msg.sender] = true;

        emit Participated(msg.sender);
    }

    function setMerkleRoot(bytes32 _merkleRoot) external onlyOwner {
        merkleRoot = _merkleRoot;
        emit MerkleRootSet(_merkleRoot);
    }

    function claimReward(
        uint256 rank,
        bytes32[] calldata proof
    ) external override nonReentrant {
        require(merkleRoot != bytes32(0), "Rewards not ready");
        require(!hasClaimed[msg.sender], "Already claimed");
        require(rank >= 1 && rank <= 3, "Invalid rank");

        // Leaf: keccak256(abi.encodePacked(account, rank))
        bytes32 leaf = keccak256(abi.encodePacked(msg.sender, rank));
        require(MerkleProof.verify(proof, merkleRoot, leaf), "Invalid proof");

        uint256 rewardAmount = (totalPool * rewardPercentages[rank - 1]) / 100;

        hasClaimed[msg.sender] = true;
        require(
            metaThreeCoin.transfer(msg.sender, rewardAmount),
            "Transfer failed"
        );

        emit RewardClaimed(msg.sender, rank, rewardAmount);
    }
}
