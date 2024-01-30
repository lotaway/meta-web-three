// SPDX-License-Identifier: MIT

pragma solidity ^0.8.20;

import "./BIERC20Factory.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

import "hardhat/console.sol";

contract BIERC20 is ERC20, Pausable, Ownable {
    uint256 public limit;
    uint256 public burns;
    uint256 public fee;
    address public factory;
    uint256 public maxMint;
    string public tick;

    uint256 public current = 0;
    string public _mintInscription = "";

    event InscribeMint(address indexed from, string content);
    event InscribeTransfer(address indexed from, string content);

    modifier notContract() {
        require(tx.origin == msg.sender);
        _;
    }

    constructor() ERC20("", "") Ownable(_msgSender()) {
        string memory _tick;
        (_tick, limit, maxMint, burns, fee) = BIERC20Factory(msg.sender).parameters();
        tick = _tick;
        factory = msg.sender;
        _mintInscription = string.concat('data:,{"p":"lins20","op":"mint","tick":"', tick, '","amt":"', Strings.toString(limit/(10 ** decimals())), '"}');
    }

    function symbol() public view override returns (string memory) {
        return tick;
    }

    function name() public view override returns (string memory) {
        return string.concat("inscription ", tick);
    }

    function decimals() public pure override returns (uint8) {
        return 18;
    }

    receive() external payable {
        _doMint();
    }

    function mint() external payable {
        _doMint();
    }

    function mintV2(string memory content)  external payable {
        require(Strings.equal(content, _mintInscription), "inscription incorrect");
        _doMint();
    }

    function _doMint() internal whenNotPaused notContract {
        require(msg.value >= fee, "fee not enough");
        require(limit + current <= maxMint, "mint over");

        _mint(msg.sender, limit);
        current += limit;
        emit InscribeMint(msg.sender, _mintInscription);
    }

    function transfer(address to, uint256 amount) public override whenNotPaused returns (bool) {
        uint256 destory = 0;
        if(burns != 0) {
            destory = Math.mulDiv(amount, burns, 10000);
        }
        require(balanceOf(msg.sender) >= (destory + amount), "insufficient balance");

        if(destory != 0) {
            _burn(msg.sender, destory);
        }
        _transfer(msg.sender, to, amount);
        
        uint256 denominator = 10 ** decimals();
        uint256 fraction = amount % denominator;
        uint256 integer  = amount / denominator;
        string memory value = Strings.toString(integer);
        if(fraction != 0) {
           fraction = fraction / (10 ** (decimals() - 4));
           value = string.concat(value, ".", Strings.toString(fraction));
        }
        string memory ins = string.concat('data:,{"p":"lins20","op":"transfer","tick":"', tick, '","amt":"', value, '","to":"', Strings.toHexString(to), '"}');
        emit InscribeTransfer(msg.sender, ins);
        return true;
    }

    function withdraw() public onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
}
