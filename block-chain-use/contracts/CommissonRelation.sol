//  SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import "./interface/ICommissionRelation.sol";

contract CommissionRelation is ICommissionRelation, Ownable {
    uint8 public constant MAX_LEVEL = 7; // 最大层级数
    
    // 存储上级关系
    mapping(address => address) public uplines;
    // 存储每个地址的层级数
    mapping(address => uint8) public levels;
    // 存储每个地址的下级数量
    mapping(address => uint256) public downlineCount;
    
    mapping(address => bool) public authorizedMinters;
    
    event UplineSet(address indexed account, address indexed upline);
    event MinterAuthorized(address indexed minter);
    event MinterRevoked(address indexed minter);
    
    modifier onlyAuthorizedMinter() {
        require(authorizedMinters[msg.sender] || msg.sender == owner(), "Not authorized");
        _;
    }
    
    constructor() Ownable(msg.sender) {}
    
    // 设置上级关系
    function setUpline(address account, address upline) external onlyAuthorizedMinter {
        require(account != address(0), "Invalid account");
        require(upline != address(0), "Invalid upline");
        require(account != upline, "Cannot set self as upline");
        require(uplines[account] == address(0), "Upline already set");
        
        // 检查是否形成循环引用
        address currentUpline = upline;
        for(uint8 i = 0; i < MAX_LEVEL; i++) {
            require(currentUpline != account, "Circular reference detected");
            if(currentUpline == address(0)) break;
            currentUpline = uplines[currentUpline];
        }
        
        // 设置上级关系
        uplines[account] = upline;
        downlineCount[upline]++;
        
        // 计算并更新层级
        updateLevels(account);
        
        emit UplineSet(account, upline);
    }
    
    // 更新层级信息
    function updateLevels(address account) internal {
        address current = account;
        uint8 level = 1;
        
        while(uplines[current] != address(0) && level <= MAX_LEVEL) {
            current = uplines[current];
            levels[current] = level;
            level++;
        }
    }
    
    // 获取某个地址的所有上级
    function getUplines(address account) external view returns (address[] memory) {
        address[] memory result = new address[](MAX_LEVEL);
        address current = account;
        uint8 count = 0;
        
        while(uplines[current] != address(0) && count < MAX_LEVEL) {
            current = uplines[current];
            result[count] = current;
            count++;
        }
        
        // 创建适当大小的数组
        address[] memory trimmedResult = new address[](count);
        for(uint8 i = 0; i < count; i++) {
            trimmedResult[i] = result[i];
        }
        
        return trimmedResult;
    }
    
    // 检查地址是否为有效的分销商
    function isValidDistributor(address distributor) external view returns (bool) {
        return distributor != address(0) && levels[distributor] > 0;
    }
    
    // 获取某个地址的层级
    function getLevel(address account) external view returns (uint8) {
        return levels[account];
    }
    
    // 获取某个地址的直接上级
    function getUpline(address account) external view returns (address) {
        return uplines[account];
    }
    
    // 获取某个地址的下级数量
    function getDownlineCount(address account) external view returns (uint256) {
        return downlineCount[account];
    }
    
    function authorizeMinter(address minter) external onlyOwner {
        authorizedMinters[minter] = true;
        emit MinterAuthorized(minter);
    }
    
    function revokeMinter(address minter) external onlyOwner {
        authorizedMinters[minter] = false;
        emit MinterRevoked(minter);
    }
}
