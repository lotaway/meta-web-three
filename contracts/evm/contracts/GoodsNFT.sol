// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interface/ICommissionToken.sol";
import "./interface/ICommissionRelation.sol";
import "./struct/GoodsSpecification.sol";
import "./interface/IGoodsNFT.sol";
contract GoodsNFT is IGoodsNFT, ERC721, Ownable, ReentrancyGuard {
    IERC20 public metaThreeCoin;
    ICommissionToken public commissionToken;
    ICommissionRelation public commissionRelation;

    mapping(uint256 => Good) public goods;
    mapping(uint256 => uint256) public goodPrices;
    uint256 private _nextTokenId;

    uint256 public constant RATE_DENOMINATOR = 100;

    mapping(string => bytes32) public couponBatchRoots;
    mapping(bytes32 => bool) public usedCoupons;

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
        uint256 commission,
        uint256 finalPrice
    );
    event CouponBatchRootSet(string batchId, bytes32 root);
    event CouponUsed(
        string indexed batchId,
        string couponCode,
        address indexed buyer
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

    function setCouponBatchRoot(
        string calldata batchId,
        bytes32 root
    ) external onlyOwner {
        couponBatchRoots[batchId] = root;
        emit CouponBatchRootSet(batchId, root);
    }

    function buy(uint256 tokenId, address referrer) external {
        mint(tokenId, referrer);
    }

    function buyWithCoupon(
        uint256 tokenId,
        address referrer,
        string calldata batchId,
        uint256 discount,
        uint256 minPrice,
        uint256 startTime,
        uint256 endTime,
        string calldata couponCode,
        bytes32[] calldata proof
    ) external nonReentrant {
        bytes32 root = couponBatchRoots[batchId];
        require(root != bytes32(0), "Batch root not set");
        require(block.timestamp <= endTime, "Coupon expired");

        bytes32 leaf = keccak256(
            abi.encodePacked(
                msg.sender,
                couponCode,
                discount,
                minPrice,
                startTime,
                endTime
            )
        );
        require(MerkleProof.verify(proof, root, leaf), "Invalid coupon proof");
        require(!usedCoupons[leaf], "Coupon already used");

        _processPurchase(tokenId, referrer, finalPrice);
        emit CouponUsed(batchId, couponCode, msg.sender);
    }

    function mint(uint256 tokenId, address referrer) public nonReentrant {
        require(_exists(tokenId), "Good does not exist");
        uint256 price = goodPrices[tokenId];
        _processPurchase(tokenId, referrer, price);
    }

    function _processPurchase(
        uint256 tokenId,
        address referrer,
        uint256 finalPrice
    ) internal {
        _takePayment(finalPrice);
        uint256 commission = (finalPrice * COMMISSION_RATE) / RATE_DENOMINATOR;
        _handleReferral(referrer, commission);
        _safeMint(msg.sender, tokenId);
        emit GoodMinted(tokenId, msg.sender, referrer, commission, finalPrice);
    }

    function _takePayment(uint256 amount) private {
        require(
            metaThreeCoin.transferFrom(msg.sender, address(this), amount),
            "Payment failed"
        );
    }

    function _handleReferral(address referrer, uint256 commission) private {
        if (referrer == address(0)) return;
        require(
            commissionRelation.isValidDistributor(referrer),
            "Invalid referrer"
        );
        if (commissionRelation.getUpline(msg.sender) == address(0)) {
            commissionRelation.setUpline(msg.sender, referrer);
        }
        require(
            commissionToken.mint(msg.sender, commission),
            "Commission mint failed"
        );
        _rewardUplines(commission);
    }

    function _rewardUplines(uint256 commission) private {
        address[] memory uplines = commissionRelation.getUplines(msg.sender);
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
