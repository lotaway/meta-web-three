//  SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

contract AccountTemplate {

    struct Sale {
        address seller;
        address contractAddress;
    }
    mapping(address => Sale) public shopping;

    /*function getSales() public view return(mapping) {
        return shopping;
    }*/

    /*function getSale(address buyerAddress) public view {
        return shopping[buyerAddress];
    }*/
}
