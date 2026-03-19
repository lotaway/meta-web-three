pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title AgentProtocol (ERC-8004 / Trustless Agents)
 * @dev A public registry and trust layer for autonomous AI agents.
 * Implements Agent Discovery, Identity, and Reputation.
 */
contract AgentProtocol is ERC721URIStorage, Ownable, ReentrancyGuard {
    struct Agent {
        uint256 agentId;
        address owner;
        string agentURI; // Linked to metadata (IPFS/Arweave)
        string[] capabilities; // On-chain tags (e.g., ["search", "summarize"])
        bool active;
    }

    struct Reputation {
        uint256 score;
        uint256 totalJobs;
    }

    struct Validation {
        address validator;
        bool approved;
    }

    struct Job {
        address requester;
        uint256 agentId;
        string inputURI;
        string outputURI;
        bool completed;
        bool validated;
    }

    uint256 private _nextTokenId;
    uint256 public jobCount;

    mapping(uint256 => Agent) public agents;
    mapping(uint256 => Reputation) public reputations; // Linked to agentId
    mapping(uint256 => Job) public jobs;
    mapping(uint256 => Validation[]) public validations;

    event AgentRegistered(
        uint256 indexed agentId,
        address indexed owner,
        string agentURI
    );
    event AgentUpdated(
        uint256 indexed agentId,
        string agentURI,
        string[] capabilities
    );
    event JobCreated(
        uint256 indexed jobId,
        uint256 indexed agentId,
        address requester
    );
    event JobCompleted(uint256 indexed jobId);
    event JobValidated(uint256 indexed jobId, bool success);

    constructor() ERC721("Agent Identity", "AGENT") Ownable(msg.sender) {}

    /**
     * @dev Registers a new agent. Mints an identity NFT to the owner.
     * @param agentURI The URI to the metadata JSON (IPFS/Arweave).
     * @param capabilities A list of capability tags for on-chain discovery.
     */
    function registerAgent(
        string calldata agentURI,
        string[] calldata capabilities
    ) external nonReentrant returns (uint256) {
        uint256 agentId = ++_nextTokenId;

        _safeMint(msg.sender, agentId);
        _setTokenURI(agentId, agentURI);

        Agent storage agent = agents[agentId];
        agent.agentId = agentId;
        agent.owner = msg.sender;
        agent.agentURI = agentURI;
        agent.active = true;

        for (uint256 i = 0; i < capabilities.length; i++) {
            agent.capabilities.push(capabilities[i]);
        }

        // Initialize reputation for the agent identity
        reputations[agentId] = Reputation({score: 0, totalJobs: 0});

        emit AgentRegistered(agentId, msg.sender, agentURI);
        return agentId;
    }

    /**
     * @dev Updates agent metadata and capabilities. Only original owner can update.
     */
    function updateAgent(
        uint256 agentId,
        string calldata agentURI,
        string[] calldata capabilities,
        bool active
    ) external {
        require(ownerOf(agentId) == msg.sender, "Not the agent owner");

        Agent storage agent = agents[agentId];
        agent.agentURI = agentURI;
        agent.active = active;

        // Clear old capabilities and update with new ones
        delete agent.capabilities;
        for (uint256 i = 0; i < capabilities.length; i++) {
            agent.capabilities.push(capabilities[i]);
        }

        _setTokenURI(agentId, agentURI);

        emit AgentUpdated(agentId, agentURI, capabilities);
    }

    /**
     * @dev Creates a job request for a specific agent.
     */
    function createJob(uint256 agentId, string calldata inputURI) external {
        require(agents[agentId].active, "Agent is not active");

        uint256 jobId = ++jobCount;

        jobs[jobId] = Job({
            requester: msg.sender,
            agentId: agentId,
            inputURI: inputURI,
            outputURI: "",
            completed: false,
            validated: false
        });

        emit JobCreated(jobId, agentId, msg.sender);
    }

    /**
     * @dev Agent submits work output.
     */
    function completeJob(
        uint256 jobId,
        string calldata outputURI
    ) external nonReentrant {
        Job storage job = jobs[jobId];

        require(
            ownerOf(job.agentId) == msg.sender,
            "Only the agent owner can complete"
        );
        require(!job.completed, "Job already completed");

        job.outputURI = outputURI;
        job.completed = true;

        emit JobCompleted(jobId);
    }

    /**
     * @dev Submit validation for a completed job.
     */
    function submitValidation(uint256 jobId, bool approved) external {
        require(jobs[jobId].completed, "Job not completed");
        require(!jobs[jobId].validated, "Job already validated");

        validations[jobId].push(
            Validation({validator: msg.sender, approved: approved})
        );
    }

    /**
     * @dev Finalize job and update agent reputation based on validations.
     */
    function finalizeJob(uint256 jobId) external {
        Job storage job = jobs[jobId];
        require(job.completed, "Job not completed");
        require(!job.validated, "Job already validated");

        uint256 approveCount;
        uint256 totalValidations = validations[jobId].length;

        for (uint256 i = 0; i < totalValidations; i++) {
            if (validations[jobId][i].approved) {
                approveCount++;
            }
        }

        bool success = false;
        if (totalValidations > 0 && approveCount > (totalValidations / 2)) {
            reputations[job.agentId].score += 1;
            success = true;
        }

        reputations[job.agentId].totalJobs += 1;
        job.validated = true;

        emit JobValidated(jobId, success);
    }

    /**
     * @dev Helper to get agent capabilities.
     */
    function getCapabilities(
        uint256 agentId
    ) external view returns (string[] memory) {
        return agents[agentId].capabilities;
    }

    // Overrides required by Solidity.
    function tokenURI(
        uint256 tokenId
    ) public view override(ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(
        bytes4 interfaceId
    ) public view override(ERC721URIStorage) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}
