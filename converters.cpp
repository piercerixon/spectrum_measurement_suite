//Bonus converters if current method is too slow
/*
void setup_converters(std::complex<short>* re) {
		
	//Set up the converters
	uhd::convert::id_type in_id;
    in_id.input_format = "sc16"; //This is actually be format ... T.T sc16_item32_be
    in_id.num_inputs = 1;
    in_id.output_format = "sc16_item32_le";
	in_id.num_outputs = 1;

	uhd::convert::id_type out_id;
    out_id.input_format = "sc16_item32_le";
    out_id.num_inputs = 1;
    out_id.output_format = "fc32";
    out_id.num_outputs = 1;

	uhd::convert::converter::sptr c0 = uhd::convert::get_converter(in_id)();
    c0->set_scalar(32767.);

	uhd::convert::converter::sptr c1 = uhd::convert::get_converter(out_id)();
	c1->set_scalar(1/32767.);

	std::vector<sc16_t> input(WIN_SAMPS);
	std::vector<boost::uint32_t> interm(WIN_SAMPS);
    std::vector<fc32_t> output(WIN_SAMPS);

	std::vector<const void *> input0(1, &re[0]), input1(1, &interm[0]);
    std::vector<void *> output0(1, &interm[0]), output1(1, &output[0]);
}
*/