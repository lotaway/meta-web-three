package com.metawebthree.finance.interfaces.facade.asset;

import com.metawebthree.finance.application.command.asset.dto.AssetDisposalCreateCommand;
import com.metawebthree.finance.application.command.asset.dto.AssetInventoryCreateCommand;
import com.metawebthree.finance.application.command.asset.dto.DepreciationGenerateCommand;
import com.metawebthree.finance.application.command.asset.dto.FixedAssetCreateCommand;
import com.metawebthree.finance.application.command.asset.FixedAssetCommandService;
import com.metawebthree.finance.application.query.asset.FixedAssetQueryService;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDepreciationDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDisposalDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetInventoryDO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/fixed-asset")
public class FixedAssetController {
    private final FixedAssetCommandService commandService;
    private final FixedAssetQueryService queryService;

    public FixedAssetController(FixedAssetCommandService commandService, FixedAssetQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping
    public ResponseEntity<IdResponse> createAsset(@RequestBody FixedAssetCreateCommand command) {
        Long id = commandService.createAsset(command);
        return ResponseEntity.ok(new IdResponse(id));
    }

    @PutMapping("/{id}")
    public ResponseEntity<Void> updateAsset(@PathVariable Long id, @RequestBody FixedAssetCreateCommand command) {
        command.setId(id);
        commandService.updateAsset(command);
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteAsset(@PathVariable Long id) {
        commandService.deleteAsset(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{id}")
    public ResponseEntity<FixedAssetDO> getAsset(@PathVariable Long id) {
        FixedAssetDO asset = queryService.getAssetById(id);
        return ResponseEntity.ok(asset);
    }

    @GetMapping("/code/{code}")
    public ResponseEntity<FixedAssetDO> getAssetByCode(@PathVariable String code) {
        FixedAssetDO asset = queryService.getAssetByCode(code);
        return ResponseEntity.ok(asset);
    }

    @GetMapping("/list")
    public ResponseEntity<List<FixedAssetDO>> listAssets(
            @RequestParam(required = false) Long departmentId,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String category) {
        List<FixedAssetDO> assets;
        if (departmentId != null) {
            assets = queryService.listAssetsByDepartment(departmentId);
        } else if (status != null) {
            assets = queryService.listAssetsByStatus(status);
        } else if (category != null) {
            assets = queryService.listAssetsByCategory(category);
        } else {
            assets = queryService.listAllAssets();
        }
        return ResponseEntity.ok(assets);
    }

    @GetMapping("/statistics")
    public ResponseEntity<FixedAssetQueryService.AssetStatistics> getAssetStatistics() {
        FixedAssetQueryService.AssetStatistics stats = queryService.getAssetStatistics();
        return ResponseEntity.ok(stats);
    }

    @PostMapping("/transfer/{id}")
    public ResponseEntity<Void> transferAsset(
            @PathVariable Long id,
            @RequestBody TransferRequest request) {
        commandService.transferAsset(id, request.getNewDepartmentId(), request.getNewDepartmentName(), 
            request.getNewLocation(), request.getNewCustodian());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/depreciation/generate")
    public ResponseEntity<Void> generateDepreciation(@RequestBody DepreciationGenerateCommand command) {
        commandService.generateDepreciation(command);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{id}/depreciation")
    public ResponseEntity<List<FixedAssetDepreciationDO>> getAssetDepreciation(@PathVariable Long id) {
        List<FixedAssetDepreciationDO> depreciationList = queryService.listDepreciationByAssetId(id);
        return ResponseEntity.ok(depreciationList);
    }

    @GetMapping("/depreciation/list")
    public ResponseEntity<List<FixedAssetDepreciationDO>> listDepreciationByPeriod(@RequestParam String period) {
        List<FixedAssetDepreciationDO> depreciationList = queryService.listDepreciationByPeriod(period);
        return ResponseEntity.ok(depreciationList);
    }

    @GetMapping("/depreciation/statistics")
    public ResponseEntity<FixedAssetQueryService.DepreciationStatistics> getDepreciationStatistics(@RequestParam String period) {
        FixedAssetQueryService.DepreciationStatistics stats = queryService.getDepreciationStatistics(period);
        return ResponseEntity.ok(stats);
    }

    @PostMapping("/inventory")
    public ResponseEntity<IdResponse> createInventory(@RequestBody AssetInventoryCreateCommand command) {
        Long id = commandService.createInventory(command);
        return ResponseEntity.ok(new IdResponse(id));
    }

    @PostMapping("/inventory/{id}/confirm")
    public ResponseEntity<Void> confirmInventory(
            @PathVariable Long id,
            @RequestBody ConfirmInventoryRequest request) {
        commandService.confirmInventory(id, request.getHandleResult());
        return ResponseEntity.ok().build();
    }

    @GetMapping("/inventory/list")
    public ResponseEntity<List<FixedAssetInventoryDO>> listInventory(@RequestParam(required = false) String status) {
        List<FixedAssetInventoryDO> inventoryList = queryService.listInventoryByStatus(status);
        return ResponseEntity.ok(inventoryList);
    }

    @GetMapping("/inventory/statistics")
    public ResponseEntity<FixedAssetQueryService.InventoryStatistics> getInventoryStatistics() {
        FixedAssetQueryService.InventoryStatistics stats = queryService.getInventoryStatistics();
        return ResponseEntity.ok(stats);
    }

    @PostMapping("/disposal")
    public ResponseEntity<IdResponse> createDisposal(@RequestBody AssetDisposalCreateCommand command) {
        Long id = commandService.createDisposal(command);
        return ResponseEntity.ok(new IdResponse(id));
    }

    @PostMapping("/disposal/{id}/approve")
    public ResponseEntity<Void> approveDisposal(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.approveDisposal(id, request.getApproverId(), request.getApproverName(), request.getComment());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/disposal/{id}/reject")
    public ResponseEntity<Void> rejectDisposal(
            @PathVariable Long id,
            @RequestBody ApproveRequest request) {
        commandService.rejectDisposal(id, request.getApproverId(), request.getApproverName(), request.getComment());
        return ResponseEntity.ok().build();
    }

    @GetMapping("/disposal/list")
    public ResponseEntity<List<FixedAssetDisposalDO>> listDisposal(@RequestParam(required = false) String status) {
        List<FixedAssetDisposalDO> disposalList = queryService.listDisposalByStatus(status);
        return ResponseEntity.ok(disposalList);
    }

    public record IdResponse(Long id) {}

    public static class TransferRequest {
        private Long newDepartmentId;
        private String newDepartmentName;
        private String newLocation;
        private String newCustodian;

        public Long getNewDepartmentId() { return newDepartmentId; }
        public String getNewDepartmentName() { return newDepartmentName; }
        public String getNewLocation() { return newLocation; }
        public String getNewCustodian() { return newCustodian; }

        public void setNewDepartmentId(Long newDepartmentId) { this.newDepartmentId = newDepartmentId; }
        public void setNewDepartmentName(String newDepartmentName) { this.newDepartmentName = newDepartmentName; }
        public void setNewLocation(String newLocation) { this.newLocation = newLocation; }
        public void setNewCustodian(String newCustodian) { this.newCustodian = newCustodian; }
    }

    public static class ConfirmInventoryRequest {
        private String handleResult;

        public String getHandleResult() { return handleResult; }
        public void setHandleResult(String handleResult) { this.handleResult = handleResult; }
    }

    public static class ApproveRequest {
        private Long approverId;
        private String approverName;
        private String comment;

        public Long getApproverId() { return approverId; }
        public String getApproverName() { return approverName; }
        public String getComment() { return comment; }

        public void setApproverId(Long approverId) { this.approverId = approverId; }
        public void setApproverName(String approverName) { this.approverName = approverName; }
        public void setComment(String comment) { this.comment = comment; }
    }
}