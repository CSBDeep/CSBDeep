#!/bin/sh

OUTPUTDIR=/home/uschmidt/research/csbdeep/CSBDeep_code/docs/build/html

# FTP_HOST=ftp.uweschmidt.org
# FTP_USER=111766-27-uwe
# FTP_TARGET_DIR=/webseiten/csbdeep.uweschmidt.org
# lftp ftp://$FTP_USER@$FTP_HOST -e "set ssl:verify-certificate no ; mirror -R $OUTPUTDIR $FTP_TARGET_DIR ; quit"

FTP_HOST=home18440447.1and1-data.host
FTP_USER=p6831244-fjug
FTP_TARGET_DIR=/doc
lftp sftp://$FTP_USER@$FTP_HOST -e "set sftp:auto-confirm yes ; mirror -R $OUTPUTDIR $FTP_TARGET_DIR ; quit"
