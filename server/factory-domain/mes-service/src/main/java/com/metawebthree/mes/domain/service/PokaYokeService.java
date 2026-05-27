package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.PokaYokeRule;
import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.entity.ProcessParameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class PokaYokeService {

    public record PokaYokeResult(
        boolean passed,
        String ruleCode,
        String message,
        PokaYokeRule.CheckAction.ActionType actionType
    ) {}

    public record MaterialCheckContext(
        String scannedMaterialCode,
        String expectedMaterialCode,
        String materialBatchNo,
        ProductionTask task
    ) {}

    public record SequenceCheckContext(
        ProductionTask currentTask,
        Integer expectedStepNo,
        Integer actualStepNo
    ) {}

    public record ParameterCheckContext(
        String parameterCode,
        Double recordedValue,
        Double standardValue,
        Double upperLimit,
        Double lowerLimit,
        ProductionTask task
    ) {}

    public List<PokaYokeResult> checkMaterial(MaterialCheckContext ctx) {
        List<PokaYokeResult> results = new ArrayList<>();
        
        if (ctx.scannedMaterialCode() == null || ctx.expectedMaterialCode() == null) {
            return results;
        }

        boolean materialMatched = ctx.scannedMaterialCode().equals(ctx.expectedMaterialCode());
        
        if (!materialMatched) {
            results.add(new PokaYokeResult(
                false,
                "MATERIAL_MISMATCH",
                "物料不匹配: 扫描物料[" + ctx.scannedMaterialCode() + 
                "]与BOM指定物料[" + ctx.expectedMaterialCode() + "]不符",
                PokaYokeRule.CheckAction.ActionType.BLOCK
            ));
        }

        return results;
    }

    public List<PokaYokeResult> checkSequence(SequenceCheckContext ctx) {
        List<PokaYokeResult> results = new ArrayList<>();
        
        if (ctx.expectedStepNo() == null || ctx.actualStepNo() == null) {
            return results;
        }

        if (ctx.actualStepNo() > ctx.expectedStepNo() + 1) {
            results.add(new PokaYokeResult(
                false,
                "SEQUENCE_SKIP",
                "跳序报警: 当前工序[" + ctx.actualStepNo() + 
                "]跳过了应为的工序[" + (ctx.expectedStepNo() + 1) + "]",
                PokaYokeRule.CheckAction.ActionType.WARNING
            ));
        } else if (ctx.actualStepNo() < ctx.expectedStepNo()) {
            results.add(new PokaYokeResult(
                false,
                "SEQUENCE_BACKWARD",
                "逆序报警: 不允许返回到之前的工序[" + ctx.actualStepNo() + "]",
                PokaYokeRule.CheckAction.ActionType.BLOCK
            ));
        }

        return results;
    }

    public List<PokaYokeResult> checkParameter(ParameterCheckContext ctx) {
        List<PokaYokeResult> results = new ArrayList<>();
        
        if (ctx.recordedValue() == null || ctx.upperLimit() == null || ctx.lowerLimit() == null) {
            return results;
        }

        if (ctx.recordedValue() > ctx.upperLimit()) {
            results.add(new PokaYokeResult(
                false,
                "PARAMETER_UPPER_EXCEED",
                "参数超差报警: 参数[" + ctx.parameterCode() + "]值[" + ctx.recordedValue() + 
                "]超过上限[" + ctx.upperLimit() + "]",
                PokaYokeRule.CheckAction.ActionType.BLOCK
            ));
        } else if (ctx.recordedValue() < ctx.lowerLimit()) {
            results.add(new PokaYokeResult(
                false,
                "PARAMETER_LOWER_EXCEED",
                "参数超差报警: 参数[" + ctx.parameterCode() + "]值[" + ctx.recordedValue() + 
                "]低于下限[" + ctx.lowerLimit() + "]",
                PokaYokeRule.CheckAction.ActionType.BLOCK
            ));
        } else if (isNearLimit(ctx.recordedValue(), ctx.standardValue(), 
                               ctx.upperLimit(), ctx.lowerLimit())) {
            results.add(new PokaYokeResult(
                true,
                "PARAMETER_NEAR_LIMIT",
                "参数接近限值: 参数[" + ctx.parameterCode() + "]值[" + ctx.recordedValue() + 
                "]接近规格限值，请注意监控",
                PokaYokeRule.CheckAction.ActionType.WARNING
            ));
        }

        return results;
    }

    private boolean isNearLimit(Double value, Double standard, 
                                Double upper, Double lower) {
        if (value == null || standard == null || upper == null || lower == null) {
            return false;
        }
        double range = upper - lower;
        double upperWarning = upper - range * 0.1;
        double lowerWarning = lower + range * 0.1;
        return value >= upperWarning || value <= lowerWarning;
    }

    public Optional<ProcessRoute.ProcessStep> validateTaskSequence(
            ProductionTask task, ProcessRoute route) {
        
        if (task == null || route == null || route.getSteps() == null) {
            return Optional.empty();
        }

        Integer currentStepNo = getStepNoFromProcessCode(task.getProcessCode(), route);
        Integer expectedStepNo = getExpectedStepNo(task, route);
        
        if (currentStepNo == null || expectedStepNo == null) {
            return Optional.empty();
        }

        if (currentStepNo > expectedStepNo + 1) {
            return route.getStepByNo(expectedStepNo + 1);
        }
        
        return Optional.empty();
    }

    private Integer getStepNoFromProcessCode(String processCode, ProcessRoute route) {
        if (processCode == null || route.getSteps() == null) {
            return null;
        }
        return route.getSteps().stream()
            .filter(s -> processCode.equals(s.getProcessCode()))
            .map(ProcessRoute.ProcessStep::getStepNo)
            .findFirst()
            .orElse(null);
    }

    private Integer getExpectedStepNo(ProductionTask task, ProcessRoute route) {
        if (task.getStatus() == ProductionTask.TaskStatus.PENDING) {
            ProcessRoute.ProcessStep firstStep = route.getFirstStep().orElse(null);
            return firstStep != null ? firstStep.getStepNo() : null;
        }

        return route.getSteps().stream()
            .filter(s -> {
                Integer successor = s.getSuccessorStepNo();
                return successor != null && route.getStepByNo(successor)
                    .map(step -> task.getProcessCode().equals(step.getProcessCode()))
                    .orElse(false);
            })
            .map(ProcessRoute.ProcessStep::getStepNo)
            .findFirst()
            .orElse(null);
    }
}