import os
prof_id = os.environ.get('PROF_ID')


profile_images_dir=f'../aftershoot/Profile_{prof_id}/TIFFs/'
sliders_table_path=f'../aftershoot/Profile_{prof_id}/sliders_exif.csv'


if __name__ == '__main__':
    print(prof_id)
