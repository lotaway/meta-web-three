// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interface/ICommissionToken.sol";
import "./interface/ICommissionRelation.sol";
import "./struct/GoodsSpecification.sol";
import "./interface/IGoodsNFT.sol";
/**
 * 从有权限的管理者后台创建
 * 注意保持足够的费用
 * todo 需要添加更多角色
 */
contract GoodsNFT is IGoodsNFT, ERC721, Ownable {
    IERC20 public metaThreeCoin;
    ICommissionToken public commissionToken;
    ICommissionRelation public commissionRelation;

    mapping(uint256 => Good) public goods;
    mapping(uint256 => uint256) public goodPrices;
    uint256 private _nextTokenId;

    uint256 public constant COMMISSION_RATE = 10; // 10% commission
    uint256 public constant UPLINE_COMMISSION_RATE = 15; // 15% for uplines
    uint256 public constant RATE_DENOMINATOR = 100;

    event Replenishment(
        uint256 indexed tokenId,
        string name,
        uint256 price,
        address creator
    );
    event GoodMinted(
        uint256 indexed tokenId,
        address indexed buyer,
        address indexed referrer,
        uint256 commission
    );

    constructor(
        string memory name,
        string memory symbol,
        address _commissionRelation,
        address _metaThreeCoin,
        address _commissionToken
    ) ERC721(name, symbol) Ownable(msg.sender) {
        commissionRelation = ICommissionRelation(_commissionRelation);
        metaThreeCoin = IERC20(_metaThreeCoin);
        commissionToken = ICommissionToken(_commissionToken);
    }

    function replenishment(
        string memory name,
        string[] memory keys,
        string[] memory values,
        uint256 price
    ) external onlyOwner returns (uint256) {
        require(
            keys.length == values.length,
            "Keys and values length mismatch"
        );

        uint256 tokenId = _nextTokenId++;
        Good storage good = goods[tokenId];
        good.name = name;
        goodPrices[tokenId] = price;

        for (uint i = 0; i < keys.length; i++) {
            good.specifications.push(Specification(keys[i], values[i]));
        }

        emit Replenishment(tokenId, name, price, msg.sender);
        return tokenId;
    }

    function buy(uint256 tokenId, address referrer) external {
        mint(tokenId, referrer);
    }

    function mint(uint256 tokenId, address referrer) public nonReentrant {
        require(_exists(tokenId), "Good does not exist");
        uint256 price = goodPrices[tokenId];

        // Transfer MetaThreeCoin from buyer
        require(
            metaThreeCoin.transferFrom(msg.sender, address(this), price),
            "Payment failed"
        );

        // Calculate commission
        uint256 commission = (price * COMMISSION_RATE) / RATE_DENOMINATOR;

        // Set up commission relationship and distribute commissions
        if (referrer != address(0)) {
            // Get approval from CommissionRelation contract first
            require(
                commissionRelation.isValidDistributor(referrer),
                "Invalid referrer"
            );

            // Only set upline if not already set
            if (commissionRelation.getUpline(msg.sender) == address(0)) {
                commissionRelation.setUpline(msg.sender, referrer);
            }

            // Distribute commission to buyer
            require(
                commissionToken.mint(msg.sender, commission),
                "Commission mint failed"
            );

            // Distribute commission to uplines
            address[] memory uplines = commissionRelation.getUplines(
                msg.sender
            );
            uint256 uplineCommission = (commission * UPLINE_COMMISSION_RATE) /
                RATE_DENOMINATOR;

            for (uint i = 0; i < uplines.length; i++) {
                if (uplines[i] == address(0)) break;
                require(
                    commissionToken.mint(uplines[i], uplineCommission),
                    "Upline commission mint failed"
                );
            }
        }

        _safeMint(msg.sender, tokenId);
        emit GoodMinted(tokenId, msg.sender, referrer, commission);
    }

    function getGoodSpecifications(
        uint256 tokenId
    ) external view returns (Specification[] memory) {
        require(_exists(tokenId), "Good does not exist");
        return goods[tokenId].specifications;
    }

    function _exists(uint256 tokenId) internal view returns (bool) {
        return bytes(goods[tokenId].name).length > 0;
    }
}
