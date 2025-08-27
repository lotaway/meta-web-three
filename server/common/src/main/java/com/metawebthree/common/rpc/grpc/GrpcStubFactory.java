package com.metawebthree.common.rpc.grpc;

import com.metawebthree.common.generated.rpc.RiskScorerServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@RequiredArgsConstructor
public class GrpcStubFactory {

    private final ZkEndpointResolver resolver;

    public RiskScorerServiceGrpc.RiskScorerServiceBlockingStub riskScorerBlockingStub() {
        return blockingStub("com.metawebthree.common.generated.rpc.RiskScorerService");
    }

    public RiskScorerServiceGrpc.RiskScorerServiceBlockingStub blockingStub(String serviceName) {
        List<ZkEndpointResolver.HostAndPort> endpoints = resolver.resolve(serviceName);
        ZkEndpointResolver.HostAndPort ep = resolver.pickOne(endpoints);
        if (ep == null) {
            throw new IllegalStateException("No available endpoints for service: " + serviceName);
        }
        ManagedChannel channel = ManagedChannelBuilder.forAddress(ep.host(), ep.port())
                .usePlaintext()
                .build();
        return RiskScorerServiceGrpc.newBlockingStub(channel);
    }
}
