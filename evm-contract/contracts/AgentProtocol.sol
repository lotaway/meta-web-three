pragma solidity ^0.8.20;

contract AgentProtocol {
    struct Agent {
        address owner;
        string metadataURI;
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
        address agent;
        string inputURI;
        string outputURI;
        bool completed;
        bool validated;
    }

    uint256 public agentCount;
    uint256 public jobCount;

    mapping(uint256 => Agent) public agents;
    mapping(address => Reputation) public reputations;
    mapping(uint256 => Job) public jobs;
    mapping(uint256 => Validation[]) public validations;

    event AgentRegistered(uint256 id, address owner);
    event JobCreated(uint256 jobId);
    event JobCompleted(uint256 jobId);
    event JobValidated(uint256 jobId);

    function registerAgent(string calldata metadataURI) external {
        agentCount++;

        agents[agentCount] = Agent({
            owner: msg.sender,
            metadataURI: metadataURI,
            active: true
        });

        reputations[msg.sender] = Reputation({score: 0, totalJobs: 0});

        emit AgentRegistered(agentCount, msg.sender);
    }

    function createJob(address agent, string calldata inputURI) external {
        jobCount++;

        jobs[jobCount] = Job({
            requester: msg.sender,
            agent: agent,
            inputURI: inputURI,
            outputURI: "",
            completed: false,
            validated: false
        });

        emit JobCreated(jobCount);
    }

    function completeJob(uint256 jobId, string calldata outputURI) external {
        Job storage job = jobs[jobId];

        require(msg.sender == job.agent);
        require(!job.completed);

        job.outputURI = outputURI;
        job.completed = true;

        emit JobCompleted(jobId);
    }

    function submitValidation(uint256 jobId, bool approved) external {
        require(jobs[jobId].completed);

        validations[jobId].push(
            Validation({validator: msg.sender, approved: approved})
        );
    }

    function finalizeJob(uint256 jobId) external {
        Job storage job = jobs[jobId];
        require(job.completed);
        require(!job.validated);

        uint256 approveCount;

        for (uint256 i = 0; i < validations[jobId].length; i++) {
            if (validations[jobId][i].approved) {
                approveCount++;
            }
        }

        if (approveCount > 0) {
            reputations[job.agent].score += 1;
            reputations[job.agent].totalJobs += 1;
        }

        job.validated = true;

        emit JobValidated(jobId);
    }
}
