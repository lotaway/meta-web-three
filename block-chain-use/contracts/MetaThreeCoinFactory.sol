//  SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/Ownable2StepUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/extensions/AccessControlDefaultAdminRulesUpgradeable.sol";

contract MetaThreeChinFactory is Initializable, AccessControlDefaultAdminRulesUpgradeable, ReentrancyGuardUpgradeable, UUPSUpgradeable {

    bytes32 private PREFIX = keccak256("meta");

    bytes32 private DOMAIN_SEPARATOR;
    bytes32 private ABI_HASH = keccak256("balanceOf(address addr) public returns(uint256)");

    constructor() {
        
    }

    function initialize() public initializer {

        uint256 id;
        assembly {
            id := chainid()
        }
        DOMAIN_SEPARATOR = keccak256(abi.encode(ABI_HASH, PREFIX, keccak256(bytes('1')), id, address(this)));

        __AccessControl_init();
        __AccessControlDefaultAdminRules_init(3 days ,_msgSender());
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
    }

}