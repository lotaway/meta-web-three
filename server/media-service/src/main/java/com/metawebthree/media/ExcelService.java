package com.metawebthree.media;

import com.alibaba.excel.EasyExcel;
import com.alibaba.excel.context.AnalysisContext;
import com.alibaba.excel.event.AnalysisEventListener;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.cloud.DefaultS3Config;
import com.metawebthree.common.cloud.DefaultS3Service;
import com.metawebthree.media.BO.ExcelTemplateBO;
import com.metawebthree.media.DO.ArtWorkDO;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class ExcelService {

    private final StringRedisTemplate redisTemplate;
    private static final String IMPORT_EXCEL_QUEUE_KEY = "excel-processing-queue";
    private static final String FAILED_DECODE_IMPORT_EXCEL_QUEUE_KEY = "failed-decode-excel-processing-queue";
    private static final String FAILED_ENCODE_IMPORT_EXCEL_QUEUE_KEY = "failed-encode-excel-processing-queue";
    private final ArtWorkMapper artworkMapper;
    private final ArtWorkCategoryMapper artworkCategoryMapper;
    private final PeopleMapper peopleMapper;
    private final PeopleTypeMapper peopleTypeMapper;
    private final ArtWorkTagMapper artworkTagMapper;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 1000;
    private static final int PROCESSING_THREADS = 2;
    private final DefaultS3Service s3Service;
    private final DefaultS3Config s3Config;

    private final ExecutorService processingExecutor = Executors.newFixedThreadPool(PROCESSING_THREADS);
    private final ExecutorService insertionExecutor = Executors.newSingleThreadExecutor();

    private interface FieldSetter<T> {
        void setField(String value, T entity);
    }

    private final Map<String, FieldSetter<ExcelTemplateBO>> FIELD_SETTERS = new LinkedHashMap<>();

    @PostConstruct
    private void initialize() {
        FIELD_SETTERS.put("作品ID", (String value, ExcelTemplateBO entity) -> entity.setId(value));
        FIELD_SETTERS.put("标题", (String value, ExcelTemplateBO entity) -> entity.setTitle(value));
        FIELD_SETTERS.put("副标题", (String value, ExcelTemplateBO entity) -> entity.setSubtitle(value));
        FIELD_SETTERS.put("季数", (String value, ExcelTemplateBO entity) -> entity.setSeason(value));
        FIELD_SETTERS.put("集数", (String value, ExcelTemplateBO entity) -> entity.setEpisode(value));
        FIELD_SETTERS.put("类别", (String value, ExcelTemplateBO entity) -> entity.setCategoryName(value));
        FIELD_SETTERS.put("标签", (String value, ExcelTemplateBO entity) -> entity.setTagNames(value));
        FIELD_SETTERS.put("年份标签", (String value, ExcelTemplateBO entity) -> entity.setYearTag(value));
        FIELD_SETTERS.put("演员", (String value, ExcelTemplateBO entity) -> entity.setActNames(value));
        FIELD_SETTERS.put("导演", (String value, ExcelTemplateBO entity) -> entity.setDirectorName(value));
        FIELD_SETTERS.put("封面链接", (String value, ExcelTemplateBO entity) -> entity.setCover(value));
        FIELD_SETTERS.put("详情链接", (String value, ExcelTemplateBO entity) -> entity.setLink(value));
    }

    public void processExcelData(String excelUrl) throws RuntimeException {
        processExcelData(excelUrl, MAX_BATCH_SIZE);
    }

    public void processExcelData(String excelUrl, int batchSize) throws RuntimeException {
        // if (!excelUrl.matches("(?i).*\\.xlsx?$")) {
        // throw new IllegalArgumentException("URL must point to a downloadable Excel
        // file (.xls or .xlsx)");
        // }

        try (InputStream inputStream = new URI(excelUrl).toURL().openStream()) {
            String contentType = new URI(excelUrl).toURL().openConnection().getContentType();
            if (contentType != null && !contentType.toLowerCase().contains("excel")
                    && !contentType.toLowerCase().contains("spreadsheet")) {
                throw new IllegalArgumentException("URL must point to an Excel file, got content type: " + contentType);
            }

            var excelListener = new CustomExcelListener(Math.max(MIN_BATCH_SIZE, Math.min(batchSize, MAX_BATCH_SIZE)));
            EasyExcel.read(inputStream, excelListener).sheet().doRead();
            excelListener.completionLatchAwait();
            log.info("Excel data processing completed successfully");
        } catch (Exception e) {
            String errorMsg = "Failed to process Excel data from URL: " + excelUrl;
            if (excelUrl.contains("docs.qq.com")) {
                errorMsg += "\nNote: Online document sharing links are not supported. Please provide a direct download link to the Excel file.";
            }
            log.error(errorMsg, e);
            throw new RuntimeException(errorMsg, e);
        }
    }

    private class CustomExcelListener extends AnalysisEventListener<Map<Integer, String>> {
        private final int batchSize;
        private final List<Map<Integer, String>> rawDataBuffer = new ArrayList<>();
        private Map<Integer, FieldSetter<ExcelTemplateBO>> columnMapping;
        private final CountDownLatch completionLatch = new CountDownLatch(1);
        private static final String END_MARKER = "__END__";
        private final ObjectMapper objectMapper = new ObjectMapper();

        public CustomExcelListener(int batchSize) {
            this.batchSize = batchSize;
            insertionExecutor.execute(() -> {
                while (!Thread.currentThread().isInterrupted()) {
                    List<String> datas = redisTemplate.opsForList().leftPop(IMPORT_EXCEL_QUEUE_KEY,
                            batchSize);
                    boolean isEnd = datas.isEmpty() || datas.get(datas.size() - 1).equals(END_MARKER);
                    List<ArtWorkDO> batch = datas.stream().filter(data -> !data.equals(END_MARKER)).map(data -> {
                        try {
                            return objectMapper.readValue(data, ArtWorkDO.class);
                        } catch (JsonProcessingException e) {
                            redisTemplate.opsForList().rightPush(FAILED_DECODE_IMPORT_EXCEL_QUEUE_KEY, data);
                            log.error("Failed to deserialize JSON to ArtWorkDO", e);
                            return null;
                        }
                    }).filter(Objects::nonNull).toList();
                    if (!batch.isEmpty()) {
                        artworkMapper.insert(batch);
                    }
                    if (isEnd) {
                        completionLatch.countDown();
                        break;
                    }
                }
            });
        }

        public boolean completionLatchAwait() throws InterruptedException {
            return this.completionLatch.await(5, TimeUnit.MINUTES);
        }

        @Override
        public void invokeHeadMap(Map<Integer, String> headMap, AnalysisContext context) {
            columnMapping = new HashMap<>();
            headMap.forEach((index, header) -> {
                var fieldSetter = FIELD_SETTERS.get(header);
                if (fieldSetter == null) {
                    return;
                }
                columnMapping.put(index, fieldSetter);
            });
        }

        @Override
        public void invoke(Map<Integer, String> data, AnalysisContext context) {
            if (columnMapping == null || columnMapping.isEmpty()) {
                String msg = "columnMapping for head is not initialized, maybe invokeHeadMap not invoked";
                log.error(msg, context);
                throw new IllegalStateException(msg);
            }

            rawDataBuffer.add(data);
            if (rawDataBuffer.size() >= batchSize) {
                processBatchInBackground(new ArrayList<>(rawDataBuffer));
                rawDataBuffer.clear();
            }
        }

        private void processBatchInBackground(List<Map<Integer, String>> batchData) {
            processingExecutor.submit(() -> {
                List<ArtWorkDO> processedBatch = new ArrayList<>();
                for (Map<Integer, String> data : batchData) {
                    var excelTemplateBO = new ExcelTemplateBO();
                    var artWorkDO = new ArtWorkDO();

                    data.forEach((index, value) -> {
                        FieldSetter<ExcelTemplateBO> fieldSetter = columnMapping.get(index);
                        if (fieldSetter == null || value == null || value.isEmpty()) {
                            return;
                        }
                        fieldSetter.setField(value, excelTemplateBO);
                    });
                    artWorkDO.setId(null);
                    artWorkDO.setTitle(excelTemplateBO.getTitle());
                    artWorkDO.setCover(excelTemplateBO.getCover());
                    artWorkDO.setLink(excelTemplateBO.getLink());
                    artWorkDO.setSubtitle(excelTemplateBO.getSubtitle());
                    artWorkDO.setSeason(excelTemplateBO.getSeasonValue());
                    artWorkDO.setEpisode(excelTemplateBO.getEpisodeValue());
                    artWorkDO.setCategoryId(excelTemplateBO.updateCategoryNameToCategoryId(artworkCategoryMapper));
                    artWorkDO.setTags(excelTemplateBO.updateTagNamesToTagIds(artworkTagMapper).toArray(new Integer[0]));
                    artWorkDO.setActs(excelTemplateBO.updateActNamesToActIds(peopleMapper, peopleTypeMapper)
                            .toArray(new Integer[0]));
                    artWorkDO.setYearTag(excelTemplateBO.getYear());

                    processedBatch.add(artWorkDO);
                }
                var artWorkRedisTemplate = new RedisTemplate<String, ArtWorkDO>()
                        .boundListOps(FAILED_ENCODE_IMPORT_EXCEL_QUEUE_KEY);
                redisTemplate.opsForList().rightPushAll(IMPORT_EXCEL_QUEUE_KEY,
                        processedBatch.stream().map(data -> {
                            try {
                                return objectMapper.writeValueAsString(data);
                            } catch (JsonProcessingException e) {
                                log.error("Failed to serialize batch to JSON", e);
                                artWorkRedisTemplate.rightPush(data);
                                return null;
                            }
                        }).filter(Objects::nonNull).toList());
            });
        }

        @Override
        public void doAfterAllAnalysed(AnalysisContext context) {
            if (!rawDataBuffer.isEmpty()) {
                processBatchInBackground(new ArrayList<>(rawDataBuffer));
                rawDataBuffer.clear();
            }
            try {
                processingExecutor.shutdown();
                if (!processingExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                    processingExecutor.shutdownNow();
                }
                redisTemplate.opsForList().rightPush(IMPORT_EXCEL_QUEUE_KEY, END_MARKER);
                insertionExecutor.shutdown();
                if (!insertionExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                    insertionExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.error("Interrupted while waiting for executor shutdown", e);
            }
        }
    }

    @PreDestroy
    public void onDestroy() {
        processingExecutor.shutdown();
        insertionExecutor.shutdown();
        try {
            if (!processingExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                processingExecutor.shutdownNow();
            }
            if (!insertionExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                insertionExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            processingExecutor.shutdownNow();
            insertionExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public String generateTemplate() {
        String CACHE_TEMPLATE_PATH = "/excel/cache/template.xlsx";
        Optional<String> cachedTemplate = s3Service.getFileUrlWithCheck(s3Config.getName(), CACHE_TEMPLATE_PATH);
        if (cachedTemplate.isPresent()) {
            return cachedTemplate.get();
        }
        try (ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            List<String> headers = new ArrayList<>(FIELD_SETTERS.keySet());
            EasyExcel.write(outputStream)
                    .head(headers.stream().map(Collections::singletonList).toList())
                    .sheet("template")
                    .doWrite(generateExampleData());
            byte[] fileContent = outputStream.toByteArray();
            String fileName = URLEncoder.encode("template.xlsx", StandardCharsets.UTF_8.toString())
                    .replaceAll("\\+", "%20");
            return s3Service.uploadExcel(s3Config.getName(), fileContent, fileName, true);
        } catch (Exception e) {
            log.error("Failed to generate Excel template", e);
            throw new RuntimeException("Failed to generate Excel template", e);
        }
    }

    protected List<List<Object>> generateExampleData() {
        List<Object> exampleRow1 = Arrays.asList(
                "",
                "示例作品2023",
                "https://example.com/cover1.jpg",
                "https://example.com/detail1",
                "示例副标题",
                1,
                10,
                "类别如Movie",
                "标签1,标签2",
                "2023",
                "演员1,演员2",
                "导演1");
        List<Object> exampleRow2 = Arrays.asList(
                "",
                "超人联盟",
                "https://youku.com/gis/superman_legend.jpg",
                "https://youku.com/video/superman_legend",
                "假期大作战",
                1,
                1,
                "Movie",
                "喜剧,超能力,动画",
                "2022",
                "凯奇·古丽",
                "雷蒙·利特波");
        return Arrays.asList(exampleRow1, exampleRow2);
    }
}
