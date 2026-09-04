package com.metawebthree.recommendation.application.aishopping;

import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import org.springframework.stereotype.Component;

/** Guards AI shopping operations behind the global enable switch. */
@Component
public class AiShoppingFeatureGuard {

    private final AiProviderConfig providerConfig;

    public AiShoppingFeatureGuard(AiProviderConfig providerConfig) {
        this.providerConfig = providerConfig;
    }

    public void requireEnabled() {
        if (!providerConfig.isEnabled()) {
            throw new BusinessException(ResponseStatus.FORBIDDEN, "AI shopping is disabled");
        }
    }
}
