// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

/**
* It has to be a chairman create this vote and send the proposal list;
* Only chairman can give the vote right to people;
* One people can only vote once, including chairman;
* People can delegate to other to help him vote, even him has a delegated too;
* No one can vote to himself.
 */

 contract Ballot {

    enum VoterStatus {
        NoRight,
        HasTicker,
        Voted,
        Ban
    }

    struct Voter {
        VoterStatus status;
        uint ticker;
        address[] delegateFrom;
        address delegateTo;
        uint delegateTicker;
    }

    struct Proposal {
        bytes32 name;
        address addr;
        uint ticker;
    }

    mapping(address => Voter) public voters;

    Proposal[] public proposals;

    constructor(bytes32[] memory _proposals, address _addr) {
        for (uint i = 0; i < _proposals.length; i++) {
            proposals.push(Proposal({
                name: _proposals[i],
                addr: _addr,
                ticker: 0
            }));
        }
    }

    //  @notice need to get index in client side
    function voteToIndex(uint toIndex) external {
        Voter memory voter = voters[msg.sender];
        require(voter.status == VoterStatus.HasTicker);
        proposals[toIndex].ticker = voter.ticker + voter.delegateTicker;
    }

 }