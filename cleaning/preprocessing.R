library("RSQLite")
library(XML)
library(tm)
library(topicmodels)
library(tidyr)
library(chron)
library(dplyr)
library(readr)
library(syuzhet)
library(tictoc)

#########################################################################################################
#                                              Data Read-in
#########################################################################################################

#-----------------------------------------------first batch----------------------------------------------
bb = read_csv('billboard_lyrics_1964-2015.csv')

#only want song name, artist name, and lyrics
bb = bb[, c(2, 3, 5)]
names(bb) = c('song', 'artist', 'lyrics')

#-----------------------------------------------second batch---------------------------------------------
sd_1 = read_csv('songdata.csv')

#only want song name, artist name, and lyrics
sd_1 = sd_1[, c(1, 2, 4)]

#-----------------------------------------------third batch---------------------------------------------
sd_2_1 = read_csv('Lyrics1.csv')
names(sd_2_1) = c('song', 'artist', 'lyrics2')

sd_2_2 = read_csv('Lyrics2.csv')
names(sd_2_2) = c('song', 'artist', 'lyrics2')

#combine third batch
sd_2 = rbind(sd_2_1, sd_2_2)
rm(sd_2_1)
rm(sd_2_2)

#########################################################################################################
#                                              Data Cleaning
#########################################################################################################

tic()

removeSpecial = function(x) gsub("[^a-zA-Z0-9' -]", ' ', x)
removeNoSpace = function(x) gsub("['-]", '', x)

#convert to VCorpus
bb_corp = VCorpus(DataframeSource(bb))
sd1_corp = VCorpus(DataframeSource(sd_1))
sd2_corp = VCorpus(DataframeSource(sd_2))

#remove all special characters
bb_corp = tm_map(bb_corp, removeSpecial)
sd1_corp = tm_map(sd1_corp, removeSpecial)
sd2_corp = tm_map(sd2_corp, removeSpecial)

bb_corp = tm_map(bb_corp, removeNoSpace)
sd1_corp = tm_map(sd1_corp, removeNoSpace)
sd2_corp = tm_map(sd2_corp, removeNoSpace)

#force back into the correct datatype 
bb_corp = tm_map(bb_corp, PlainTextDocument)
sd1_corp = tm_map(sd1_corp, PlainTextDocument)
sd2_corp = tm_map(sd2_corp, PlainTextDocument)

#convert to lowercase
bb_corp = tm_map(bb_corp, content_transformer(tolower))
sd1_corp = tm_map(sd1_corp, content_transformer(tolower))
sd2_corp = tm_map(sd2_corp, content_transformer(tolower))

#remove stopwords
# bb_corp = tm_map(bb_corp, removeWords, stopwords("english"))
# sd1_corp = tm_map(sd1_corp, removeWords, stopwords("english"))
# sd2_corp = tm_map(sd2_corp, removeWords, stopwords("english"))

#strip whitespace
bb_corp = tm_map(bb_corp, stripWhitespace) 
sd1_corp = tm_map(sd1_corp, stripWhitespace) 
sd2_corp = tm_map(sd2_corp, stripWhitespace) 

toc()

#output clean
bb_out <- t(data.frame(text = sapply(bb_corp, identity), stringsAsFactors = F))
rownames(bb_out) = c()
bb_content = data.frame(content = unlist(as.data.frame(bb_out)$content))
bb_final = data.frame(title = bb_content$content[seq(1, nrow(bb_content), 3)], 
                      artist = bb_content$content[seq(2, nrow(bb_content), 3)], 
                      lyrics = bb_content$content[seq(3, nrow(bb_content), 3)])

write_csv(bb_final, 'billboard_clean.csv')

sd1_out <- t(data.frame(text = sapply(sd1_corp, identity), stringsAsFactors = F))
rownames(sd1_out) = c()
sd1_content = data.frame(content = unlist(as.data.frame(sd1_out)$content))
sd1_final = data.frame(artist = sd1_content$content[seq(1, nrow(sd1_content), 3)], 
                      title = sd1_content$content[seq(2, nrow(sd1_content), 3)], 
                      lyrics = sd1_content$content[seq(3, nrow(sd1_content), 3)])

write_csv(sd1_final, 'lyrics_1_clean.csv')

sd2_out <- t(data.frame(text = sapply(sd2_corp, identity), stringsAsFactors = F))
rownames(sd2_out) = c()
sd2_content = data.frame(content = unlist(as.data.frame(sd2_out)$content))
sd2_final = data.frame(artist = sd2_content$content[seq(1, nrow(sd2_content), 3)], 
                       title = sd2_content$content[seq(2, nrow(sd2_content), 3)], 
                       lyrics = sd2_content$content[seq(3, nrow(sd2_content), 3)])

write_csv(sd2_final, 'lyrics_2_clean.csv')

#########################################################################################################
#                                              Data Joining
#########################################################################################################

bb = read_csv('billboard_clean.csv')
names(bb) = c('title', 'artist', 'lyrics1')
bb$chart = 1

sd_1 = read_csv('lyrics_1_clean.csv')
names(sd_1) = c('artist', 'title', 'lyrics2')

sd_2 = read_csv('lyrics_2_clean.csv')
names(sd_2) = c('artist', 'lyrics3', 'title')

#merge first two datasets
join_1 = merge(bb, sd_1, by = c('title', 'artist'), all.x = T, all.y = T)

#get lyrics from either column
join_1$full_lyrics = NA
join_1$full_lyrics[is.na(join_1$lyrics1) == T] = join_1$lyrics2[is.na(join_1$lyrics1) == T]
join_1$full_lyrics[is.na(join_1$full_lyrics) == T] = join_1$lyrics1[is.na(join_1$full_lyrics) == T]

join_1 = join_1[, c(1, 2, 4, 6)]

#merge next dataset
full = merge(join_1, sd_2, by = c('title', 'artist'), all.x = T, all.y = T)

#get new lyrics
full$full_lyrics[is.na(full$full_lyrics) == T] = full$lyrics3[is.na(full$full_lyrics) == T]

full = full[, -5]

#properly assign chart variable
full$chart[is.na(full$chart) == T] = 0
full$chart = as.factor(full$chart)

#remove songs without lyrics
full = na.omit(full)

write_csv(full, 'full_lyrics.csv')