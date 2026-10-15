package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.domain.entity.*;
import com.metawebthree.mes.domain.service.AndonService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import java.util.List;

@RestController
@RequestMapping("/api/mes/andon")
public class AndonController {
    private final AndonService service;
    public AndonController(AndonService service) { this.service=service; }
    public record TypeRequest(@NotBlank String typeCode,@NotBlank String typeName,@NotNull AndonType.AndonCategory category) {}
    public record LevelRequest(@NotBlank String levelCode,@NotBlank String levelName,@NotNull @Positive Integer levelValue) {}
    public record EventRequest(@NotNull @Positive Long andonTypeId,@NotNull @Positive Long andonLevelId,
            @NotNull AndonEvent.TriggerMethod triggerMethod,@NotBlank String workshopId,@Positive Long workstationId,
            String equipmentId,@NotBlank String reporterId,@NotBlank String reporterName,@NotBlank String description) {}
    public record HandlerRequest(@NotBlank String handlerId,@NotBlank String handlerName) {}
    @PostMapping("/types") public AndonType createType(@Valid @RequestBody TypeRequest r) { return service.createAndonType(r.typeCode(),r.typeName(),r.category()); }
    @GetMapping("/types") public List<AndonType> types() { return service.getAllActiveAndonTypes(); }
    @GetMapping("/types/{id}") public AndonType type(@PathVariable Long id) { return service.getAndonTypeById(id).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND)); }
    @PostMapping("/levels") public AndonLevel createLevel(@Valid @RequestBody LevelRequest r) { return service.createAndonLevel(r.levelCode(),r.levelName(),r.levelValue()); }
    @GetMapping("/levels") public List<AndonLevel> levels() { return service.getAllActiveAndonLevels(); }
    @GetMapping("/levels/{id}") public AndonLevel level(@PathVariable Long id) { return service.getAndonLevelById(id).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND)); }
    @PostMapping("/events") public AndonEvent trigger(@Valid @RequestBody EventRequest r) {
        return service.triggerAndon(r.andonTypeId(),r.andonLevelId(),r.triggerMethod().name(),r.workshopId(),
                r.workstationId()==null?null:r.workstationId().toString(),r.equipmentId(),r.reporterId(),r.reporterName(),r.description());
    }
    @GetMapping("/events") public List<AndonEvent> events(@RequestParam(defaultValue="PENDING") AndonEvent.AndonEventStatus status) { return service.getAndonEventsByStatus(status); }
    @GetMapping("/events/{id}") public AndonEvent event(@PathVariable Long id) { return service.getAndonEventById(id).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND)); }
    @PostMapping("/events/{id}/acknowledge") public AndonEvent acknowledge(@PathVariable Long id,@Valid @RequestBody HandlerRequest r) { service.acknowledgeAndonEvent(id,r.handlerId(),r.handlerName()); return event(id); }
    @PostMapping("/events/{id}/handle") public AndonEvent handle(@PathVariable Long id) { service.startHandling(id); return event(id); }
    @PostMapping("/events/{id}/resolve") public AndonEvent resolve(@PathVariable Long id) { service.resolveAndonEvent(id); return event(id); }
    @PostMapping("/events/{id}/close") public AndonEvent close(@PathVariable Long id) { service.closeAndonEvent(id); return event(id); }
    @PostMapping("/events/{id}/escalate") public AndonEvent escalate(@PathVariable Long id) { service.escalateAndonEvent(id); return event(id); }
}
