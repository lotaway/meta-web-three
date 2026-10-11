package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.service.OAuthApplicationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "OAuth Applications", description = "OAuth 2.0 application management")
@RestController
@RequestMapping("/developer/oauth")
@RequiredArgsConstructor
public class OAuthApplicationController {

    private final OAuthApplicationService oauthService;

    @Operation(summary = "Register OAuth application", description = "Create a new OAuth 2.0 application")
    @PostMapping("/{developerId}")
    public ResponseEntity<OAuthAppResponse> registerApplication(
        @PathVariable String developerId,
        @Valid @RequestBody OAuthAppCreateRequest request
    ) {
        OAuthAppResponse response = oauthService.registerApplication(developerId, request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @Operation(summary = "Get OAuth application", description = "Get OAuth application details by client ID")
    @GetMapping("/{clientId}")
    public ResponseEntity<OAuthAppResponse> getApplication(@PathVariable String clientId) {
        OAuthAppResponse response = oauthService.getApplication(clientId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "List developer OAuth applications", description = "Get all OAuth applications for a developer")
    @GetMapping("/developer/{developerId}")
    public ResponseEntity<List<OAuthAppResponse>> getDeveloperApplications(@PathVariable String developerId) {
        List<OAuthAppResponse> response = oauthService.getDeveloperApplications(developerId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Update OAuth application", description = "Update OAuth application settings")
    @PutMapping("/{clientId}")
    public ResponseEntity<OAuthAppResponse> updateApplication(
        @PathVariable String clientId,
        @Valid @RequestBody OAuthAppCreateRequest request
    ) {
        OAuthAppResponse response = oauthService.updateApplication(clientId, request);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Regenerate client secret", description = "Generate a new client secret for the OAuth application")
    @PostMapping("/{clientId}/regenerate-secret")
    public ResponseEntity<OAuthAppResponse> regenerateClientSecret(@PathVariable String clientId) {
        OAuthAppResponse response = oauthService.regenerateClientSecret(clientId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Disable OAuth application", description = "Temporarily disable an OAuth application")
    @PostMapping("/{clientId}/disable")
    public ResponseEntity<OAuthAppResponse> disableApplication(@PathVariable String clientId) {
        OAuthAppResponse response = oauthService.disableApplication(clientId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Enable OAuth application", description = "Re-enable a disabled OAuth application")
    @PostMapping("/{clientId}/enable")
    public ResponseEntity<OAuthAppResponse> enableApplication(@PathVariable String clientId) {
        OAuthAppResponse response = oauthService.enableApplication(clientId);
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Delete OAuth application", description = "Permanently delete an OAuth application")
    @DeleteMapping("/{clientId}")
    public ResponseEntity<Void> deleteApplication(@PathVariable String clientId) {
        oauthService.deleteApplication(clientId);
        return ResponseEntity.noContent().build();
    }
}
