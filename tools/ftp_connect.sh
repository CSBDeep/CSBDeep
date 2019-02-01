#!/bin/bash

FTP_HOST=access763254689.webspace-data.io
FTP_USER=u95689219-uschmidt

lftp sftp://$FTP_USER@$FTP_HOST -e "set sftp:auto-confirm yes"
