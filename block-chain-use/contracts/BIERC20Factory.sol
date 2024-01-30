// SPDX-License-Identifier: MIT

pragma solidity ^0.8.20;

import "./BIERC20.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract BIERC20Factory is Ownable, Pausable {
    struct Parameters {
        string tick;
        uint256 limit;
        uint256 totalSupply;
        uint256 burns;
        uint256 fee;
    }

    Parameters public parameters;
    mapping(string => address) public inscriptions;
    event InscribeDeploy(address indexed from, string content);
    event AddInscription(string tick, address indexed addr);

    constructor() Ownable(_msgSender()) {}

    function createBIERC20(
        string memory tick,
        uint256 limit,
        uint256 totalSupply,
        uint256 burns,
        uint256 fee
    ) external whenNotPaused returns (address brc20) {
        require(burns < 10000, "burns out of range");
        require(limit < totalSupply, "limit out of range");
        require(inscriptions[tick] == address(0), "tick exists");
        require(totalSupply % limit == 0, "limit incorrect");

        parameters = Parameters({tick: tick, limit: limit, totalSupply: totalSupply, burns: burns, fee: fee});
        brc20 = address(new BIERC20{salt: keccak256(abi.encode(tick, limit, totalSupply, burns, fee))}());
        inscriptions[tick] = brc20;

        uint256 decimals = 10 ** 18;
        delete parameters;
        string memory ins = string.concat('data:,{"p":"lins20","op":"deploy","tick":"', tick, '","max":"', Strings.toString(totalSupply / decimals), '","lim":"', Strings.toString(limit / decimals), '"}');
        emit InscribeDeploy(msg.sender, ins);
    }

    /**
     * add new or exist inscription
     * @param tick Tick
     * @param addr Address
     */
    function addInscription(string memory tick, address addr) external onlyOwner {
        require(inscriptions[tick] == address(0), "tick exists");
        inscriptions[tick] = addr;
        emit AddInscription(tick, addr);
    }

    /**
     * @notice Pause (admin only)
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @notice Unpause (admin only)
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    function withdraw() public onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
}
