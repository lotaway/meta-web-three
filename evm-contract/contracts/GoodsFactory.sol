// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "./interface/IGoodsNFT.sol";

contract GoodsFactory is Initializable, OwnableUpgradeable {
    function initialize() public initializer {
        // __Ownable_init();
    }

    function createGoods(
        uint256 tokenId,
        string memory name,
        uint256 price
    ) public {
        // IGoodsNFT goodsNFT = new IGoodsNFT(name, symbol, address(this));
    }
}
