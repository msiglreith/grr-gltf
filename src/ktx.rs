use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

const FILE_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

#[derive(Debug, Copy, Clone)]
pub struct Header {
    pub format: u32,
    pub type_size: u32,
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub pixel_depth: u32,
    pub layer_count: u32,
    pub face_count: u32,
    pub level_count: u32,
    pub supercompression_scheme: u32,
}

pub struct Image {
    pub header: Header,
    pub levels: Vec<Vec<u8>>,
}

impl Image {
    pub fn new(data: &[u8]) -> anyhow::Result<Self> {
        let mut rdr = std::io::Cursor::new(data);
        let mut identifier = [0; 12];
        rdr.read_exact(&mut identifier)?;
        assert_eq!(identifier, FILE_IDENTIFIER);

        let header = Header {
            format: rdr.read_u32::<LittleEndian>()?,
            type_size: rdr.read_u32::<LittleEndian>()?,
            pixel_width: rdr.read_u32::<LittleEndian>()?,
            pixel_height: rdr.read_u32::<LittleEndian>()?,
            pixel_depth: rdr.read_u32::<LittleEndian>()?,
            layer_count: rdr.read_u32::<LittleEndian>()?,
            face_count: rdr.read_u32::<LittleEndian>()?,
            level_count: rdr.read_u32::<LittleEndian>()?,
            supercompression_scheme: rdr.read_u32::<LittleEndian>()?,
        };

        let _dfd_offset = rdr.read_u32::<LittleEndian>()?;
        let _dfd_len = rdr.read_u32::<LittleEndian>()?;

        let _kvd_offset = rdr.read_u32::<LittleEndian>()?;
        let _kvd_len = rdr.read_u32::<LittleEndian>()?;

        let _sgd_offset = rdr.read_u64::<LittleEndian>()?;
        let _sgd_len = rdr.read_u64::<LittleEndian>()?;

        let levels = (0..header.level_count.max(1))
            .map(|_| {
                let offset = rdr.read_u64::<LittleEndian>()?;
                let len = rdr.read_u64::<LittleEndian>()?;
                let len_uncompressed = rdr.read_u64::<LittleEndian>()?;

                assert_eq!(len, len_uncompressed); // TODO

                Ok(data[offset as usize..(offset + len) as usize].to_vec())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Image { header, levels })
    }
}
