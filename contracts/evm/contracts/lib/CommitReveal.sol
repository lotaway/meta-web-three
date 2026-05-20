// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CommitReveal {
    struct Commit {
        bytes32 commitHash;
        uint64 blockNumber;
        bool revealed;
    }

    uint256 public revealDeadline;

    address[] public pendingCommits;
    mapping(address => Commit) public commits;

    function commit(bytes32 _hash) public {
        require(commits[msg.sender].commitHash == 0, "Already committed");
        commits[msg.sender] = Commit(_hash, uint64(block.number), false);
        pendingCommits.push(msg.sender);
    }

    function reveal(uint256 _value, bytes32 _salt) public {
        cleanupExpired();
        Commit storage commit = commits[msg.sender];
        require(commit.commitHash != 0, "Not committed");
        require(!commit.revealed, "Already revealed");
        require(
            block.number > commit.blockNumber,
            "Cannot reveal in same block"
        );
        require(
            keccak256(abi.encodePacked(_value, _salt)) == commit.commitHash,
            "Invalid reveal"
        );

        commit.revealed = true;
        // use _value as random number or selection
    }

    function cleanupExpired() public {
        require(block.timestamp > revealDeadline);
        for (uint i = 0; i < pendingCommits.length; i++) {
            if (!commits[pendingCommits[i]].revealed) {
                delete commits[pendingCommits[i]];
            }
        }
    }

    /// @dev use users's revealed values to generate random number
    function generateRandomNumber() public view returns (uint256) {}
}
