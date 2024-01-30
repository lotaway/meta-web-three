//  SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

contract EncryptedToken {
    uint256 INIT_TOKENS = 10;
    uint256 TOKENS = 999_999;
    mapping(address => uint256) balances;

    function initToken() {
        balances[msg.sender] = INIT_TOKENS;
    }

    // （从合约内）转账给指定账户
    function transfer(address _to, uint256 _amount) {
        assert(balances[msg.sender] >= _amount);
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
    }

    // 查看账户余额
    function balanceOf(address _owner) view returns (uint256) {
        return balances[_owner];
    }
}
