// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract WETH is IERC20 {
    string public name = "Wrapped Ether";
    string public symbol = "WETH";
    uint8 public decimals = 18;

    event Deposit(address indexed sender, uint256 amount);
    event Withdrawal(address indexed receiver, uint256 amount);

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    receive() external payable {
        deposit(IERC20(address(0)), address(this), msg.value);
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function totalSupply() public view returns (uint256) {
        return address(this).balance;
    }

    function deposit(IERC20 token, address to, uint256 amount) public payable {
        if (address(token) == address(0) && amount == 0) {
            balanceOf[msg.sender] += msg.value;
            emit Deposit(msg.sender, msg.value);
        }
        else {
            require(token.balanceOf(msg.sender) >= amount, "No enough balance");
            token.transferFrom(address(token), to, amount);
            emit Deposit(msg.sender, amount);
        }
    }

    function withdraw(uint256 amount) public {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
        emit Withdrawal(msg.sender, amount);
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Insufficient allowance");
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        allowance[from][msg.sender] -= amount;
        emit Transfer(from, to, amount);
        return true;
    }

    // Additional ERC-20 events and functions (like `emit Transfer` and `emit Approval`) would be here
}