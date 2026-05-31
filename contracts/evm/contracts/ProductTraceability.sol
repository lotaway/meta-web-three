// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/governance/TimelockController.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title ProductTraceability
 * @dev Product full-chain blockchain traceability contract
 * Implements: production, transportation, sales full-process on-chain
 * Uses AccessControl for role-based permission management
 */
contract ProductTraceability is AccessControl, ReentrancyGuard {
    // Role definitions
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant PRODUCER_ROLE = keccak256("PRODUCER_ROLE");
    bytes32 public constant TRANSPORTER_ROLE = keccak256("TRANSPORTER_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    
    // Timelock controller for multi-sig admin management
    TimelockController public timelockController;
    
    // Initialization flag
    bool public initialized;
    
    uint256 private _nextTraceId;

    // Trace record status
    enum TraceStatus {
        Created,
        ProductionCompleted,
        InTransit,
        Delivered,
        Sold
    }

    // Trace record structure
    struct TraceRecord {
        uint256 traceId;
        string productId;
        string productName;
        string batchNumber;
        address producer;
        uint256 productionTime;
        TraceStatus status;
        TraceEvent[] events;
    }

    // Trace event structure
    struct TraceEvent {
        uint256 traceId;
        string eventType;
        string description;
        string location;
        address operator;
        uint256 timestamp;
        string extraData;
    }

    // Product info structure
    struct ProductInfo {
        string productId;
        string productName;
        string category;
        string manufacturer;
        string productionLocation;
        uint256 productionDate;
        bool isActive;
    }

    // Mappings
    mapping(uint256 => TraceRecord) public traceRecords;
    mapping(string => uint256[]) public productTraceIds;
    mapping(string => ProductInfo) public productInfos;
    mapping(string => bytes32[]) public productBatchHashes;

    /**
     * @dev Grant producer role to an address
     */
    function grantProducerRole(address account) external onlyAdmin {
        grantRole(PRODUCER_ROLE, account);
    }

    /**
     * @dev Grant transporter role to an address
     */
    function grantTransporterRole(address account) external onlyAdmin {
        grantRole(TRANSPORTER_ROLE, account);
    }

    /**
     * @dev Grant verifier role to an address
     */
    function grantVerifierRole(address account) external onlyAdmin {
        grantRole(VERIFIER_ROLE, account);
    }

    /**
     * @dev Revoke producer role from an address
     */
    function revokeProducerRole(address account) external onlyAdmin {
        revokeRole(PRODUCER_ROLE, account);
    }

    /**
     * @dev Revoke transporter role from an address
     */
    function revokeTransporterRole(address account) external onlyAdmin {
        revokeRole(TRANSPORTER_ROLE, account);
    }

    /**
     * @dev Check if account has producer role
     */
    function hasProducerRole(address account) external view returns (bool) {
        return hasRole(PRODUCER_ROLE, account);
    }

    /**
     * @dev Check if account has transporter role
     */
    function hasTransporterRole(address account) external view returns (bool) {
        return hasRole(TRANSPORTER_ROLE, account);
    }

    /**
     * @dev Check if account has admin role
     */
    function hasAdminRole(address account) external view returns (bool) {
        return hasRole(ADMIN_ROLE, account);
    }
    
    // Events
    event TraceCreated(
        uint256 indexed traceId,
        string productId,
        string batchNumber,
        address producer
    );
    event ProductRegistered(
        string indexed productId,
        string productName,
        address producer
    );
    event TraceEventAdded(
        uint256 indexed traceId,
        string eventType,
        address operator
    );
    event TraceStatusChanged(
        uint256 indexed traceId,
        TraceStatus oldStatus,
        TraceStatus newStatus
    );

    // Event for initialization
    event ContractInitialized(address timelockController);
    
    /**
     * @dev Initialize the contract with timelock controller
     * @param _timelock Address of the TimelockController (multi-sig)
     */
    function initialize(address _timelock) external {
        require(!initialized, "Already initialized");
        require(_timelock != address(0), "Invalid timelock address");
        
        timelockController = TimelockController(_timelock);
        
        // Grant admin role to timelock controller (not to EOA)
        _grantRole(DEFAULT_ADMIN_ROLE, _timelock);
        _grantRole(ADMIN_ROLE, _timelock);
        
        initialized = true;
        _nextTraceId = 1;
        
        emit ContractInitialized(_timelock);
    }

    // Modifiers for role-based access
    modifier onlyAdmin() {
        _checkRole(ADMIN_ROLE, msg.sender);
        _;
    }

    modifier onlyProducer() {
        _checkRole(PRODUCER_ROLE, msg.sender);
        _;
    }

    modifier onlyTransporter() {
        _checkRole(TRANSPORTER_ROLE, msg.sender);
        _;
    }

    modifier onlyVerifier() {
        _checkRole(VERIFIER_ROLE, msg.sender);
        _;
    }

    /**
     * @dev Register a new product - requires ADMIN_ROLE
     */
    function registerProduct(
        string memory productId,
        string memory productName,
        string memory category,
        string memory manufacturer,
        string memory productionLocation,
        uint256 productionDate
    ) external onlyAdmin {
        require(
            bytes(productInfos[productId].productId).length == 0,
            "Product already registered"
        );
        require(bytes(productId).length > 0, "Product ID is required");
        require(bytes(productName).length > 0, "Product name is required");

        productInfos[productId] = ProductInfo({
            productId: productId,
            productName: productName,
            category: category,
            manufacturer: manufacturer,
            productionLocation: productionLocation,
            productionDate: productionDate,
            isActive: true
        });

        emit ProductRegistered(productId, productName, msg.sender);
    }

    /**
     * @dev Create trace record for product
     */
    function createTraceRecord(
        string memory productId,
        string memory batchNumber
    ) external nonReentrant returns (uint256) {
        require(
            bytes(productInfos[productId].productId).length > 0,
            "Product not registered"
        );
        require(bytes(batchNumber).length > 0, "Batch number is required");

        uint256 traceId = _nextTraceId++;
        ProductInfo memory productInfo = productInfos[productId];

        TraceRecord storage record = traceRecords[traceId];
        record.traceId = traceId;
        record.productId = productId;
        record.productName = productInfo.productName;
        record.batchNumber = batchNumber;
        record.producer = msg.sender;
        record.productionTime = block.timestamp;
        record.status = TraceStatus.Created;

        productTraceIds[productId].push(traceId);

        emit TraceCreated(traceId, productId, batchNumber, msg.sender);
        return traceId;
    }

    /**
     * @dev Add trace event
     */
    function addTraceEvent(
        uint256 traceId,
        string memory eventType,
        string memory description,
        string memory location,
        string memory extraData
    ) external nonReentrant {
        require(
            bytes(traceRecords[traceId].productId).length > 0,
            "Trace record not found"
        );

        TraceEvent memory traceEvent = TraceEvent({
            traceId: traceId,
            eventType: eventType,
            description: description,
            location: location,
            operator: msg.sender,
            timestamp: block.timestamp,
            extraData: extraData
        });

        traceRecords[traceId].events.push(traceEvent);

        emit TraceEventAdded(traceId, eventType, msg.sender);
    }

    /**
     * @dev Update trace status
     */
    function updateTraceStatus(
        uint256 traceId,
        TraceStatus newStatus
    ) external nonReentrant {
        require(
            bytes(traceRecords[traceId].productId).length > 0,
            "Trace record not found"
        );

        TraceStatus oldStatus = traceRecords[traceId].status;
        traceRecords[traceId].status = newStatus;

        emit TraceStatusChanged(traceId, oldStatus, newStatus);
    }

    /**
     * @dev Record production completion
     */
    function recordProduction(
        uint256 traceId,
        string memory location,
        string memory qualityInfo
    ) external nonReentrant {
        require(
            traceRecords[traceId].status == TraceStatus.Created,
            "Invalid trace status"
        );

        traceRecords[traceId].status = TraceStatus.ProductionCompleted;

        TraceEvent memory traceEvent = TraceEvent({
            traceId: traceId,
            eventType: "PRODUCTION_COMPLETED",
            description: qualityInfo,
            location: location,
            operator: msg.sender,
            timestamp: block.timestamp,
            extraData: ""
        });

        traceRecords[traceId].events.push(traceEvent);

        emit TraceStatusChanged(
            traceId,
            TraceStatus.Created,
            TraceStatus.ProductionCompleted
        );
    }

    /**
     * @dev Record transportation event
     */
    function recordTransportation(
        uint256 traceId,
        string memory fromLocation,
        string memory toLocation,
        string memory carrierInfo,
        string memory transportCondition
    ) external nonReentrant {
        require(
            traceRecords[traceId].status == TraceStatus.ProductionCompleted ||
                traceRecords[traceId].status == TraceStatus.InTransit,
            "Invalid trace status for transportation"
        );

        traceRecords[traceId].status = TraceStatus.InTransit;

        string memory description = string(
            abi.encodePacked(
                "From: ",
                fromLocation,
                " To: ",
                toLocation,
                " Carrier: ",
                carrierInfo,
                " Condition: ",
                transportCondition
            )
        );

        TraceEvent memory traceEvent = TraceEvent({
            traceId: traceId,
            eventType: "TRANSPORTATION",
            description: description,
            location: toLocation,
            operator: msg.sender,
            timestamp: block.timestamp,
            extraData: transportCondition
        });

        traceRecords[traceId].events.push(traceEvent);

        emit TraceStatusChanged(
            traceId,
            traceRecords[traceId].status,
            TraceStatus.InTransit
        );
    }

    /**
     * @dev Record delivery event
     */
    function recordDelivery(
        uint256 traceId,
        string memory location,
        string memory receiverInfo
    ) external nonReentrant {
        require(
            traceRecords[traceId].status == TraceStatus.InTransit,
            "Product not in transit"
        );

        traceRecords[traceId].status = TraceStatus.Delivered;

        TraceEvent memory traceEvent = TraceEvent({
            traceId: traceId,
            eventType: "DELIVERED",
            description: receiverInfo,
            location: location,
            operator: msg.sender,
            timestamp: block.timestamp,
            extraData: ""
        });

        traceRecords[traceId].events.push(traceEvent);

        emit TraceStatusChanged(
            traceId,
            TraceStatus.InTransit,
            TraceStatus.Delivered
        );
    }

    /**
     * @dev Record sales event
     */
    function recordSale(
        uint256 traceId,
        address buyer,
        string memory saleLocation,
        uint256 price
    ) external nonReentrant {
        require(
            traceRecords[traceId].status == TraceStatus.Delivered,
            "Product not delivered"
        );
        require(buyer != address(0), "Invalid buyer address");

        traceRecords[traceId].status = TraceStatus.Sold;

        string memory description = string(
            abi.encodePacked(
                "Sold to: ",
                Strings.toHexString(buyer),
                " Price: ",
                Strings.toString(price)
            )
        );

        TraceEvent memory traceEvent = TraceEvent({
            traceId: traceId,
            eventType: "SOLD",
            description: description,
            location: saleLocation,
            operator: msg.sender,
            timestamp: block.timestamp,
            extraData: Strings.toString(price)
        });

        traceRecords[traceId].events.push(traceEvent);

        emit TraceStatusChanged(
            traceId,
            TraceStatus.Delivered,
            TraceStatus.Sold
        );
    }

    /**
     * @dev Get trace record by ID
     */
    function getTraceRecord(
        uint256 traceId
    ) external view returns (TraceRecord memory) {
        require(
            bytes(traceRecords[traceId].productId).length > 0,
            "Trace record not found"
        );
        return traceRecords[traceId];
    }

    /**
     * @dev Get all trace IDs for a product
     */
    function getProductTraceIds(
        string memory productId
    ) external view returns (uint256[] memory) {
        return productTraceIds[productId];
    }

    /**
     * @dev Get product info
     */
    function getProductInfo(
        string memory productId
    ) external view returns (ProductInfo memory) {
        require(
            bytes(productInfos[productId].productId).length > 0,
            "Product not found"
        );
        return productInfos[productId];
    }

    /**
     * @dev Get trace event count
     */
    function getTraceEventCount(
        uint256 traceId
    ) external view returns (uint256) {
        require(
            bytes(traceRecords[traceId].productId).length > 0,
            "Trace record not found"
        );
        return traceRecords[traceId].events.length;
    }

    /**
     * @dev Get trace event by index
     */
    function getTraceEvent(
        uint256 traceId,
        uint256 eventIndex
    ) external view returns (TraceEvent memory) {
        require(
            bytes(traceRecords[traceId].productId).length > 0,
            "Trace record not found"
        );
        require(
            eventIndex < traceRecords[traceId].events.length,
            "Event index out of bounds"
        );
        return traceRecords[traceId].events[eventIndex];
    }

    /**
     * @dev Verify product authenticity
     */
    function verifyProduct(
        string memory productId,
        string memory batchNumber
    ) external view returns (bool) {
        uint256[] memory traceIds = productTraceIds[productId];
        if (traceIds.length == 0) return false;

        for (uint256 i = 0; i < traceIds.length; i++) {
            if (
                keccak256(
                    abi.encodePacked(traceRecords[traceIds[i]].batchNumber)
                ) ==
                keccak256(abi.encodePacked(batchNumber))
            ) {
                return true;
            }
        }
        return false;
    }
}

// Helper library for string conversions
library Strings {
    bytes16 private constant _HEX_SYMBOLS = "0123456789abcdef";

    function toString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }

    function toHexString(
        address addr
    ) internal pure returns (string memory) {
        return toHexString(uint256(uint160(addr)));
    }

    function toHexString(
        uint256 value
    ) internal pure returns (string memory) {
        uint256 length = 0;
        uint256 temp = value;
        while (temp != 0) {
            length++;
            temp >>= 8;
        }
        return toHexStringFixed(value, length);
    }

    function toHexStringFixed(
        uint256 value,
        uint256 length
    ) internal pure returns (string memory) {
        bytes memory buffer = new bytes(2 * length + 2);
        buffer[0] = "0";
        buffer[1] = "x";
        for (uint256 i = 2 * length + 1; i > 1; i--) {
            buffer[i] = _HEX_SYMBOLS[value & 0xf];
            value >>= 4;
        }
        return string(buffer);
    }
}