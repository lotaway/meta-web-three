//  SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MetaThreeCoin is ERC20 {
    uint256 public TOKENS = 99_999;
    address public owner;

    constructor(string memory name, string memory symbol) ERC20("MetaCoinETH", "ECH") {
        owner = msg.sender;
//        totalSupply = TOKENS;
//        balanceOf[msg.sender] = TOKENS;
    }

    event SetOwner(address originOwner, address newOwner);

    function setOwner(address newOwner) public {
        require(msg.sender == owner, "Not owner");
        owner = newOwner;
        emit SetOwner(owner, newOwner);
    }

    function decimals() pure public override returns (uint8) {
        return 4;
    }
}