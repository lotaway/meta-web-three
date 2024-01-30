//  SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;
import "zeppelin-solidity/contracts/token/ERC20/StandardToken.sol";

contract MetaThreeCoin is StandardToken {
    string public name = "MetaThreeCoin";
    string public symbol = "BLC";
    uint8 public decimals = 4;
    uint256 public TOKENS = 99_999;

    function MetaThreeCoin() {
        totalSupply = TOKENS;
        balances[msg.sender] = TOKENS;
    }
}