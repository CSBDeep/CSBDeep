#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUTDIR="$DIR/build/html"

FTP_HOST=access763254689.webspace-data.io
FTP_USER=u95689219-uschmidt
FTP_TARGET_DIR=/doc

lftp sftp://$FTP_USER@$FTP_HOST -e "set sftp:auto-confirm yes ; mirror -R $OUTPUTDIR $FTP_TARGET_DIR ; quit"
