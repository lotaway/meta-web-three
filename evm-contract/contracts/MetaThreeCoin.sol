//  SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract MetaThreeCoin is ERC20, Ownable, ReentrancyGuard {
    uint256 public TOKENS = 999_999_999;

    constructor(
        string memory name, 
        string memory symbol
    ) ERC20(name, symbol) Ownable(msg.sender) {
        _mint(msg.sender, TOKENS);
    }

    function decimals() pure public override returns (uint8) {
        return 2;
    }

}