// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interface/ICommissionToken.sol";
import "./interface/ICommissionRelation.sol";

contract CommissionToken is ICommissionToken, ERC20, Ownable, ReentrancyGuard {
    ICommissionRelation public commissionRelation;
    
    modifier onlyCommissionRelation() {
        require(msg.sender == address(commissionRelation), "Only commission relation contract can call");
        _;
    }
    
    constructor(
        string memory name,
        string memory symbol,
        address _commissionRelation
    ) ERC20(name, symbol) Ownable(msg.sender) {
        require(_commissionRelation != address(0), "Invalid commission relation address");
        commissionRelation = ICommissionRelation(_commissionRelation);
    }
    
    function mint(address to, uint256 amount) external onlyCommissionRelation nonReentrant returns (bool) {
        _mint(to, amount);
        emit TokensMinted(to, amount);
        return true;
    }
    
    function burn(address from, uint256 amount) external onlyCommissionRelation nonReentrant returns (bool) {
        _burn(from, amount);
        emit TokensBurned(from, amount);
        return true;
    }
    
    event TokensMinted(address indexed to, uint256 amount);
    event TokensBurned(address indexed from, uint256 amount);
}
