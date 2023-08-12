//  SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract EncryptedToken {
    uint256 INIT_TOKENS = 10;
    uint256 TOKENS = 100000;
    mapping(address => uint256) balances;

    function initToken() {
        balances[msg.sender] = INIT_TOKENS;
    }

    function transfer(address _to, uint256 _amount) {
        assert(balances[msg.sender] >= _amount);
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
    }

    function balanceOf(address _owner) view returns(uint256) {
        return balances[_owner];
    }
}
