package com.metawebthree.media.BO;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import org.apache.poi.hpsf.Array;

import com.baomidou.mybatisplus.core.metadata.TableInfo;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.github.yulichang.base.MPJBaseMapper;
import com.github.yulichang.query.MPJLambdaQueryWrapper;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.ArtWorkTagMapper;
import com.metawebthree.media.PeopleMapper;
import com.metawebthree.media.DO.ArtWorkCategoryDO;
import com.metawebthree.media.DO.ArtWorkTagDO;
import com.metawebthree.media.DO.PeopleDO;
import com.metawebthree.media.DO.PeopleTypeDO;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class ExcelTemplateBO {
    static Integer BASIC_YEAR = 1895;
    String id;
    String title;
    String cover;
    String link;
    String subtitle;
    String season;
    String episode;
    String categoryName;
    String tagNames;
    String yearTag;
    String actNames;
    String directorName;

    TableInfo peopleTableInfo = TableInfoHelper.getTableInfo(PeopleDO.class);
    String peopleTableName = peopleTableInfo.getTableName().replace("\"", "");
    TableInfo peopleTypeTableInfo = TableInfoHelper.getTableInfo(PeopleTypeDO.class);
    String peopleTypeTableName = peopleTypeTableInfo.getTableName();

    protected String getCleanName(String name) {
        return name.trim().replaceAll("\\s+", " ");
    }

    public void setCategoryName(String name) {
        categoryName = getCleanName(name);
    }

    public void setTagNames(String names) {
        tagNames = getCleanName(names);
    }

    public Integer updateCategoryNameToCategoryId(MPJBaseMapper<ArtWorkCategoryDO> artworkCategoryMapper) {
        var wrapper = new MPJLambdaQueryWrapper<ArtWorkCategoryDO>();
        wrapper.eq(ArtWorkCategoryDO::getName, categoryName).isNull(ArtWorkCategoryDO::getId).last("limit 1");
        System.out.println("Generated SQL: " + wrapper.getSqlSelect());
        ArtWorkCategoryDO result = artworkCategoryMapper.selectOne(wrapper);
        if (result != null) {
            return result.getId();
        }
        List<ArtWorkCategoryDO> results = artworkCategoryMapper.selectList(wrapper);
        if (results != null && results.size() == 1) {
            return results.get(0).getId();
        }
        var categoryDO = ArtWorkCategoryDO.builder().name(categoryName).build();
        artworkCategoryMapper.insert(categoryDO);
        return categoryDO.getId();
    }

    public List<Integer> updateTagNamesToTagIds(ArtWorkTagMapper artworkTagMapper) {
        var nameList = List.<String>of(tagNames.split(","));
        var wrapper = new MPJLambdaQueryWrapper<ArtWorkTagDO>();
        wrapper.select(ArtWorkTagDO::getId, ArtWorkTagDO::getTag).in(ArtWorkTagDO::getTag, nameList);
        List<ArtWorkTagDO> existingDOs = artworkTagMapper.selectList(wrapper);
        var missingNames = new ArrayList<String>();
        var idList = new ArrayList<Integer>();

        Stream<ArtWorkTagDO> stream = existingDOs.stream();
        nameList.forEach((String name) -> {
            ArtWorkTagDO artWorkTagDO = stream.filter(existingDO -> existingDO.getTag().equals(name)).findFirst()
                    .orElse(null);
            if (artWorkTagDO == null) {
                missingNames.add(name);
                return;
            }
            idList.add(artWorkTagDO.getId());
        });
        List<Integer> newIds = artworkTagMapper.insertBatchThenReturnIds(missingNames);
        idList.addAll(newIds);
        return idList;
    }

    public List<Integer> updateActNamesToActIds(PeopleMapper peopleMapper,
            MPJBaseMapper<PeopleTypeDO> peopleTypeMapper) {
        var nameList = List.<String>of(actNames.split(","));
        var wrapper = new MPJLambdaWrapper<PeopleDO>();
        String PEOPLE_TYPE = "Actor";
        wrapper.select(PeopleDO::getId)
                .in(PeopleDO::getName, nameList)
                .leftJoin(PeopleTypeDO.class,
                        on -> on.apply(
                                String.format("%s.id = ANY(%s.types)", peopleTypeTableName, peopleTableName)))
                .eq(PeopleTypeDO::getType, PEOPLE_TYPE);
        List<PeopleDO> existingDOs = peopleMapper.selectJoinList(wrapper);
        List<PeopleTypeDO> typeDOs = peopleTypeMapper.selectList(new MPJLambdaWrapper<PeopleTypeDO>()
                .select(PeopleTypeDO::getId).eq(PeopleTypeDO::getType, PEOPLE_TYPE));

        Stream<PeopleDO> stream = existingDOs.stream();
        var missingPeopleDOs = new ArrayList<PeopleDO>();
        var idList = new ArrayList<Integer>();

        nameList.forEach(name -> {
            PeopleDO peopleDO = stream.filter(existingDO -> existingDO.getName().equals(name)).findFirst().orElse(null);
            if (peopleDO == null) {
                missingPeopleDOs
                        .add(PeopleDO.builder().name(name).types(new Short[] { typeDOs.get(0).getId() }).build());
                return;
            }
            idList.add(peopleDO.getId());
        });
        List<Integer> newIds = peopleMapper.insertBatchThenReturnIds(missingPeopleDOs);
        idList.addAll(newIds);
        return idList;
    }

    public Integer updateDirectorNameToDirectorId(MPJBaseMapper<PeopleDO> peopleMapper,
            MPJBaseMapper<PeopleTypeDO> peopleTypeMapper) {
        var wrapper = new MPJLambdaWrapper<PeopleDO>();
        wrapper.select(PeopleDO::getId).eq(PeopleDO::getName, directorName).leftJoin(PeopleTypeDO.class,
                on -> on.apply(String.format("%s.id = ANY(%s.types)", peopleTypeTableName, peopleTableName)))
                .eq(PeopleTypeDO::getType, "Director");
        List<PeopleDO> result = peopleMapper.selectJoinList(wrapper);
        PeopleDO peopleDO;
        if (result == null || result.isEmpty()) {
            List<PeopleTypeDO> typeDOs = peopleTypeMapper.selectList(new MPJLambdaWrapper<PeopleTypeDO>()
                    .select(PeopleTypeDO::getId).eq(PeopleTypeDO::getType, directorName));
            peopleDO = PeopleDO.builder().name(directorName).types(new Short[] { typeDOs.get(0).getId() }).build();
            peopleMapper.insert(peopleDO);
        } else {
            peopleDO = result.get(0);
        }
        return peopleDO.getId();
    }

    public Integer getSeasonValue() {
        Integer value = 1;
        if (season != null && !season.isEmpty()) {
            try {
                return Integer.parseInt(season);
            } catch (NumberFormatException e) {
                // do nothing
            }
        }
        return value;
    }

    public Integer getEpisodeValue() {
        Integer value = 1;
        if (episode != null && !episode.isEmpty()) {
            try {
                value = Integer.parseInt(episode);
            } catch (NumberFormatException e) {
                // do nothing
            }
        }
        return value;
    }

    public Integer getYear() {
        Integer value = BASIC_YEAR;
        if (yearTag != null && !yearTag.isEmpty()) {
            try {
                value = Integer.parseInt(yearTag);
            } catch (NumberFormatException e) {
                // do nothing
            }
        }
        Optional<Integer> result = getYearFromTitle(title);
        return result.isEmpty() ? value : result.get();
    }

    public static Optional<Integer> getYearFromTitle(String str) {
        if (str != null) {
            var pattern = java.util.regex.Pattern.compile("(\\d{4})$");
            var matcher = pattern.matcher(str);
            if (matcher.find()) {
                var year = Integer.parseInt(matcher.group(1));
                var currentYear = java.time.Year.now().getValue();
                if (year >= BASIC_YEAR && year <= currentYear) {
                    return Optional.of(year);
                }
            }
        }
        return Optional.empty();
    }

    public String[] getTagNameArray() {
        return tagNames != null ? tagNames.split(",") : new String[0];
    }

    public String[] getActNameArray() {
        return actNames != null ? actNames.split(",") : new String[0];
    }
}
