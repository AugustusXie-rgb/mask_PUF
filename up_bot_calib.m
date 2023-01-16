% up_bot_calib

up=imread('D:\mask PUF\data\demo_data\original\demo_2_up.tif');
bot=imread('D:\mask PUF\data\demo_data\original\demo_2_bot.tif');
up=double(up);
bot=double(bot);

up_mean=mean(up(:));
bot_mean=mean(bot(:));
up_size=size(up);
bot_size=size(bot);
max_size=max(up_size,bot_size);

up_cat=cat(1,up,up_mean*ones(max_size(1)-up_size(1),up_size(2)));
up_cat=cat(2,up_cat,up_mean*ones(max_size(1),max_size(2)-up_size(2)));
bot_cat=cat(1,bot,bot_mean*ones(max_size(1)-bot_size(1),bot_size(2)));
bot_cat=cat(2,bot_cat,bot_mean*ones(max_size(1),max_size(2)-bot_size(2)));
up_ind=ones(max_size(1),max_size(2));
% up_ind=logical(up_ind);
bot_ind=ones(max_size(1),max_size(2));
% bot_ind=logical(bot_ind);
up_ind(up_size(1)+1:max_size(1),:)=0;
up_ind(:,up_size(2)+1:max_size(2))=0;
bot_ind(bot_size(1)+1:max_size(1),:)=0;
bot_ind(:,bot_size(2)+1:max_size(2))=0;

shift=[0,0];
cc_score=sum(sum(up_cat.*bot_cat));
bot_shift=bot_cat;
ind_shift=bot_ind;
check=0;

while check==0
    check=1;
%     temp_1=circshift(bot_shift,1,1);
%     temp_2=circshift(bot_shift,-1,1);
%     temp_3=circshift(bot_shift,1,2);
%     temp_4=circshift(bot_shift,-1,2);
    if sum(sum(up_cat.*circshift(bot_shift,1,1)))>cc_score
        cc_score=sum(sum(up_cat.*circshift(bot_shift,1,1)));
        check=0;
        bot_shift=circshift(bot_shift,1,1);
        ind_shift=circshift(ind_shift,1,1);
        shift(1)=shift(1)+1;
    elseif sum(sum(up_cat.*circshift(bot_shift,-1,1)))>cc_score
        cc_score=sum(sum(up_cat.*circshift(bot_shift,-1,1)));
        check=0;
        bot_shift=circshift(bot_shift,-1,1);
        ind_shift=circshift(ind_shift,-1,1);
        shift(1)=shift(1)-1;
    elseif sum(sum(up_cat.*circshift(bot_shift,1,2)))>cc_score
        cc_score=sum(sum(up_cat.*circshift(bot_shift,1,2)));
        check=0;
        bot_shift=circshift(bot_shift,1,2);
        ind_shift=circshift(ind_shift,1,2);
        shift(2)=shift(2)+1;
    elseif sum(sum(up_cat.*circshift(bot_shift,-1,2)))>cc_score
        cc_score=sum(sum(up_cat.*circshift(bot_shift,-1,2)));
        check=0;
        bot_shift=circshift(bot_shift,-1,2);
        ind_shift=circshift(ind_shift,-1,2);
        shift(2)=shift(2)-1;
    end
    shift
end

shift

ind_total=up_ind.*ind_shift;
for i=max_size(1):-1:1
    if sum(ind_total(i,:))==0
        up_cat(i,:)=[];
        bot_shift(i,:)=[];
    end
end
for i=max_size(2):-1:1
    if sum(ind_total(:,i))==0
        up_cat(:,i)=[];
        bot_shift(:,i)=[];
    end
end

up_cat=im2uint16(up_cat/65535);
bot_shift=im2uint16(bot_shift/65535);
imwrite(up_cat,'D:\mask PUF\data\demo_data\calib\2_up.tif');
imwrite(bot_shift,'D:\mask PUF\data\demo_data\calib\2_bot.tif');


