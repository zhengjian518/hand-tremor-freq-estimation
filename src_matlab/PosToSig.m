function [signal_cell] = PosToSig(path,joint_number)
	% open the folder and get the sorted list of txts contatining positions
	txt_list = dir(path);
	sorted_list = natsortfiles({txt_list.name}); 
	len = length(sorted_list);
	signal_cell = cell(1,2);
	signal = zeros(len-2,2);
	for i = 3 : length(sorted_list)

		txt_full_path = strcat(path,'/',sorted_list{i});
		fid = fopen(txt_full_path);
		acc = textscan(, '%f %f');
		signal(i-2,1) = acc{1}(joint_number); % y coord
		signal(i-2,2) = acc{2}(joint_number); % x coord
		fclose(fid);
	end

	signal_cell{1}=signal(:,1);
	signal_cell{2}=signal(:,2);
